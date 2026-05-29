import math
import torch
import torch.distributed as dist
import numpy as np


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Polar Express degree-5 iteration with original normalization preserved.
    """

    assert G.ndim >= 2

    coeffs = [
        (8.205160414005574,  -22.90193498705605,   16.460724910180314),
        (4.066395159942775,  -2.8611540867551426,  0.5183995226694741),
        (3.9095949044379155, -2.823351735039516,   0.5250369769390025),
        (3.2855640171986153, -2.415301959635945,   0.48529406552790866),
        (2.277873287083977,  -1.619821765265441,   0.39848078704168355),
    ]

    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for a, b, c in coeffs[:steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X



def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def sgd_update(grad, param, buf, momentum, weight_decay, nesterov):
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)
    if momentum != 0:
        buf.mul_(momentum).add_(grad)
        if nesterov:
            grad = grad.add(buf, alpha=momentum)
        else:
            grad = buf
    return grad


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class MuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-1)
                group["momentum"] = group.get("momentum", 0.9)
                group["weight_decay"] = group.get("weight_decay", 1e-2)
                group["nesterov"] = group.get("nesterov", False)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = sgd_update(p.grad, p, state["momentum_buffer"], group["momentum"], group["weight_decay"], group["nesterov"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-1)
                group["momentum"] = group.get("momentum", 0.9)
                group["weight_decay"] = group.get("weight_decay", 1e-2)
                group["nesterov"] = group.get("nesterov", False)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = sgd_update(p.grad, p, state["momentum_buffer"], group["momentum"], group["weight_decay"], group["nesterov"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# -----------------------------------------------------------------------------
# Memory Muon logic from muon (10), wrapped in muon (9)-style templates


def _ensure_grad(p):
    if p.grad is None:
        p.grad = torch.zeros_like(p)
    return p.grad


def _ensure_state_tensor(state, key, like_tensor):
    if key not in state:
        state[key] = torch.zeros_like(like_tensor)
    else:
        state[key] = state[key].to(device=like_tensor.device, dtype=like_tensor.dtype)
    return state[key]


def _run_sgd_step(p, state, group):
    _ensure_grad(p)
    if len(state) == 0:
        state["momentum_buffer"] = torch.zeros_like(p)
    update = sgd_update(
        p.grad,
        p,
        state["momentum_buffer"],
        group["momentum"],
        group["weight_decay"],
        group["nesterov"],
    )
    p.add_(update, alpha=-group["lr"])


def next_power_of_two(n: int):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(X):
    n = X.size(-1)
    if n & (n - 1):
        raise ValueError(f"Hadamard transform requires power-of-two width, got {n}")
    Y = X.contiguous().reshape(-1, n)
    h = 1
    while h < n:
        Y = Y.view(-1, n // (2 * h), 2, h)
        a = Y[:, :, 0, :]
        b = Y[:, :, 1, :]
        Y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (Y / math.sqrt(n)).reshape_as(X)


def orthogonalize_update_like_muon(G, steps: int):
    assert G.ndim == 2, f"expected a 2D matrix, got {tuple(G.shape)}"
    out = zeropower_via_newtonschulz5(G, steps=steps)  # O_t: direction after final NS step
    rows, cols = G.size(0), G.size(1)
    dim_scale = 0.2 * math.sqrt(min(rows, cols))
    return out, dim_scale


class _MemoryMuonBase(torch.optim.Optimizer):
    @staticmethod
    def _validate_main_params(params):
        if not params:
            raise ValueError("MemoryMuon requires at least one main parameter.")
        for p in params:
            if not isinstance(p, torch.nn.Parameter):
                raise TypeError("MemoryMuon expects torch.nn.Parameter objects.")
            if p.ndim not in (2, 4):
                raise ValueError(f"MemoryMuon only supports 2D matrices or 4D conv kernels, got shape {tuple(p.shape)}.")

    @staticmethod
    def _matrix_view(G):
        if G.ndim == 4:
            return G.contiguous().reshape(len(G), -1)
        if G.ndim == 2:
            return G.contiguous()
        raise ValueError(f"MemoryMuon only supports 2D/4D params, got gradient with shape {tuple(G.shape)}")

    @staticmethod
    def _vectorize_gradient(G):
        if G.ndim not in (2, 4):
            raise ValueError(f"MemoryMuon only supports 2D/4D params, got gradient with shape {tuple(G.shape)}")
        return G.contiguous().reshape(-1)

    @staticmethod
    def _randn(shape, device, dtype, seed):
        if seed is None:
            return torch.randn(*shape, device=device, dtype=dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        out = torch.randn(*shape, generator=gen, device="cpu", dtype=torch.float32)
        return out.to(device=device, dtype=dtype)

    def _build_srht_state(self, input_dim, target_dim, device, dtype, param_offset):
        if target_dim >= input_dim:
            return {
                "input_dim": input_dim,
                "padded_dim": input_dim,
                "target_dim": input_dim,
                "scale": 1.0,
                "signs": None,
                "sample_idx": None,
                "identity": True,
            }

        padded_dim = next_power_of_two(input_dim)
        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(input_dim) + int(param_offset)
        if seed is None:
            signs = torch.randint(0, 2, (padded_dim,), device=device, dtype=torch.int8)
            sample_idx = torch.randperm(padded_dim, device=device)[:target_dim]
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            signs = torch.randint(0, 2, (padded_dim,), generator=gen, device="cpu", dtype=torch.int8).to(device=device)
            sample_idx = torch.randperm(padded_dim, generator=gen, device="cpu")[:target_dim].to(device=device)
        signs = signs.to(dtype=dtype).mul_(2).sub_(1)
        sample_idx, _ = torch.sort(sample_idx)
        return {
            "input_dim": input_dim,
            "padded_dim": padded_dim,
            "target_dim": target_dim,
            "scale": math.sqrt(padded_dim / target_dim),
            "signs": signs,
            "sample_idx": sample_idx,
            "identity": False,
        }

    @staticmethod
    def _move_srht_state(srht_state, device, dtype):
        if srht_state is None or srht_state.get("identity", False):
            return srht_state
        srht_state["signs"] = srht_state["signs"].to(device=device, dtype=dtype)
        srht_state["sample_idx"] = srht_state["sample_idx"].to(device=device)
        return srht_state

    @staticmethod
    def _apply_srht(x, srht_state):
        if srht_state.get("identity", False):
            return x
        input_dim = srht_state["input_dim"]
        padded_dim = srht_state["padded_dim"]
        Y = x.unsqueeze(0)
        if padded_dim != input_dim:
            Y_pad = torch.zeros(1, padded_dim, device=x.device, dtype=x.dtype)
            Y_pad[:, :input_dim] = Y
            Y = Y_pad
        Y = Y * srht_state["signs"].unsqueeze(0)
        Y = fast_walsh_hadamard_transform(Y)
        Y = Y.index_select(1, srht_state["sample_idx"])
        return (Y * srht_state["scale"]).squeeze(0)

    @staticmethod
    def _apply_srht_transpose(x_proj, srht_state):
        if srht_state.get("identity", False):
            return x_proj
        padded_dim = srht_state["padded_dim"]
        Y = torch.zeros(1, padded_dim, device=x_proj.device, dtype=x_proj.dtype)
        Y[:, srht_state["sample_idx"]] = x_proj.unsqueeze(0)
        Y = Y * srht_state["scale"]
        Y = fast_walsh_hadamard_transform(Y)
        Y = Y * srht_state["signs"].unsqueeze(0)
        return Y[:, :srht_state["input_dim"]].squeeze(0)

    def _init_centroids(self, memory_dim, device, dtype, param_offset):
        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(param_offset)
        return self._randn((self.num_centroids, memory_dim), device=device, dtype=dtype, seed=seed)

    def _kmeanspp_init_centroids(self, samples, device, dtype, param_offset):
        """Initialize centroids from warm-start samples with k-means++.

        This only performs the k-means++ seeding step, not Lloyd iterations.
        If there are fewer usable samples than centroids, remaining centroids are
        filled with the running mean of the selected warm-start samples.
        """
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        samples = samples.to(device=device, dtype=dtype)
        n, d = samples.shape
        if n == 0:
            return self._init_centroids(d, device, dtype, param_offset)

        base_seed = int(self.memory_init_seed) if self.memory_init_seed is not None else int(torch.initial_seed())
        seed = base_seed + int(param_offset) + 17
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)

        centers = torch.empty(self.num_centroids, d, device=device, dtype=dtype)
        if gen is None:
            first_idx = torch.randint(n, (1,), device=device).item()
        else:
            first_idx = torch.randint(n, (1,), generator=gen, device="cpu").item()
        centers[0].copy_(samples[first_idx])

        closest_dist_sq = torch.cdist(samples, centers[:1], p=2).squeeze(1).pow_(2)
        num_selected = 1
        for k in range(1, self.num_centroids):
            total = closest_dist_sq.sum()
            if not torch.isfinite(total) or total <= 0:
                if gen is None:
                    idx = torch.randint(n, (1,), device=device).item()
                else:
                    idx = torch.randint(n, (1,), generator=gen, device="cpu").item()
            else:
                probs = (closest_dist_sq / total).detach().float().cpu()
                idx = torch.multinomial(probs, 1, replacement=True, generator=gen).item()
            centers[k].copy_(samples[idx])
            new_dist_sq = torch.cdist(samples, centers[k:k + 1], p=2).squeeze(1).pow_(2)
            closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)
            num_selected += 1

        if num_selected < self.num_centroids:
            mean = samples.mean(dim=0, keepdim=True)
            centers[num_selected:].copy_(mean.expand(self.num_centroids - num_selected, -1))
        return centers

    def _finish_kmeanspp_warmstart(self, state, samples, param_offset):
        centroids = self._kmeanspp_init_centroids(
            samples,
            device=samples.device,
            dtype=samples.dtype,
            param_offset=param_offset,
        )
        state["memory_centroids"].copy_(centroids)
        state["memory_ema_counts"].fill_(1.0)
        state["memory_ema_sums"].copy_(centroids)
        state["memory_step"] = self.num_centroids
        state["memory_ws_initialized"] = True
        state["memory_ws_buffer"] = []

    def _ensure_param_memory_state(self, p, state, grad_vec):
        device = grad_vec.device
        dtype = grad_vec.dtype
        input_dim = grad_vec.numel()
        memory_dim = min(self.centroid_dim, input_dim)

        _ensure_state_tensor(state, "momentum_buffer", p)
        if "momentum_buffer_proj" not in state:
            state["momentum_buffer_proj"] = torch.zeros(memory_dim, device=device, dtype=dtype)
        else:
            state["momentum_buffer_proj"] = state["momentum_buffer_proj"].to(device=device, dtype=dtype)

        needs_reinit = (
            "memory_centroids" not in state
            or state.get("memory_input_dim") != input_dim
            or state.get("memory_dim") != memory_dim
        )
        if needs_reinit:
            state["memory_input_dim"] = input_dim
            state["memory_dim"] = memory_dim
            state["memory_pi"] = self._build_srht_state(input_dim, memory_dim, device, dtype, id(p))
            state["memory_centroids"] = self._init_centroids(memory_dim, device, dtype, id(p))
            state["memory_step"] = 0
            state["memory_m_sum"] = torch.zeros(memory_dim, device=device, dtype=dtype)
            state["memory_ema_counts"] = torch.zeros(self.num_centroids, device=device, dtype=dtype)
            state["memory_ema_sums"] = torch.zeros(self.num_centroids, memory_dim, device=device, dtype=dtype)
            state["memory_pending_z"] = []
            state["memory_ws_buffer"] = []
            state["memory_ws_initialized"] = self.ema_ws_steps <= 0
            return

        state["memory_centroids"] = state["memory_centroids"].to(device=device, dtype=dtype)
        state["memory_m_sum"] = state["memory_m_sum"].to(device=device, dtype=dtype)
        state["memory_ema_counts"] = state["memory_ema_counts"].to(device=device, dtype=dtype)
        state["memory_ema_sums"] = state["memory_ema_sums"].to(device=device, dtype=dtype)
        state["memory_pi"] = self._move_srht_state(state["memory_pi"], device, dtype)
        state["memory_pending_z"] = [z.to(device=device, dtype=dtype) for z in state.get("memory_pending_z", [])]
        state["memory_ws_buffer"] = [z.to(device=device, dtype=dtype) for z in state.get("memory_ws_buffer", [])]
        state["memory_ws_initialized"] = state.get("memory_ws_initialized", self.ema_ws_steps <= 0)

    def _project_vector(self, vec, state):
        return self._apply_srht(vec, state["memory_pi"])

    @staticmethod
    def _warmstart_memory(z, state, num_centroids):
        step = state["memory_step"]
        state["memory_centroids"][step].copy_(z)
        state["memory_m_sum"].add_(z)
        mean = state["memory_m_sum"] / (step + 1)
        if step + 1 < num_centroids:
            state["memory_centroids"][step + 1:].copy_(mean.unsqueeze(0))
        state["memory_step"] = step + 1
        if state["memory_step"] == num_centroids:
            state["memory_ema_counts"].fill_(1.0)
            state["memory_ema_sums"].copy_(state["memory_centroids"])

    def _update_memory_batch(self, samples, state, param_offset=0):
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        if self.ema_ws_steps > 0 and not state.get("memory_ws_initialized", False):
            state["memory_ws_buffer"].extend([z.detach().clone() for z in samples])
            if len(state["memory_ws_buffer"]) < self.ema_ws_steps:
                return
            ws_samples = torch.stack(state["memory_ws_buffer"], dim=0)
            self._finish_kmeanspp_warmstart(state, ws_samples, param_offset)
            return

        if state["memory_step"] < self.num_centroids:
            remaining = self.num_centroids - state["memory_step"]
            warm_count = min(remaining, samples.size(0))
            for i in range(warm_count):
                self._warmstart_memory(samples[i], state, self.num_centroids)
            if warm_count == samples.size(0):
                return
            samples = samples[warm_count:]

        centroids = state["memory_centroids"]
        distances = torch.cdist(samples, centroids, p=2)
        winners = distances.argmin(dim=1)
        counts = torch.bincount(winners, minlength=self.num_centroids).to(device=samples.device, dtype=samples.dtype)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, winners, samples)
        gamma = self.ema_decay
        state["memory_ema_counts"].mul_(gamma).add_(counts, alpha=1.0 - gamma)
        state["memory_ema_sums"].mul_(gamma).add_(sums, alpha=1.0 - gamma)
        denom = state["memory_ema_counts"].unsqueeze(1).clamp_min(self.ema_eps)
        state["memory_centroids"].copy_(state["memory_ema_sums"] / denom)

    @staticmethod
    def _queue_memory_point(z, state):
        state["memory_pending_z"].append(z.detach().clone())

    def _flush_pending_memory(self, state, force=False, param_offset=0):
        pending = state.get("memory_pending_z", [])
        if not pending:
            return False
        if not force and len(pending) < self.step_update:
            return False
        samples = torch.stack(pending, dim=0)
        state["memory_pending_z"] = []
        self._update_memory_batch(samples, state, param_offset=param_offset)
        return True

    def _memory_correction(self, z_ns, state, ns_steps, out_dtype):
        basis = torch.cat([state["memory_centroids"], z_ns.unsqueeze(0)], dim=0)
        basis_ortho = zeropower_via_newtonschulz5(basis, steps=ns_steps)
        alpha = (basis * basis_ortho).sum() / (basis.pow(2).sum() + 1e-8)
        correction_proj = (basis_ortho[-1] - alpha * basis[-1]).to(out_dtype)
        return self._apply_srht_transpose(correction_proj, state["memory_pi"])

    def _step_muon_param(self, p, state, group):
        grad = _ensure_grad(p)
        grad_vec = self._vectorize_gradient(grad)
        self._ensure_param_memory_state(p, state, grad_vec)

        z_grad = self._project_vector(grad_vec, state)

        beta = group["momentum"]

        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - beta)  # m_t = beta * m_{t-1} + (1 - beta) * G_t
        buf_proj = state["momentum_buffer_proj"]
        buf_proj.lerp_(z_grad, 1 - beta)  # 	ilde m_t = beta * 	ilde m_{t-1} + (1 - beta) * z_t

        if group["nesterov"]:
            update_pre_ns = torch.lerp(grad, buf, beta)  # U_t = (1 - beta) * G_t + beta * m_t
            z_for_correction = torch.lerp(z_grad, buf_proj, beta)
        else:
            update_pre_ns = buf
            z_for_correction = buf_proj

        if state.get("memory_ws_initialized", False):
            correction_vec = self._memory_correction(z_for_correction, state, group["ns_steps"], grad_vec.dtype)
            correction = correction_vec.view_as(grad)
            update_in = update_pre_ns + self.lambda_memory * correction
        else:
            # During k-means++ warm start, do not use random centroids for memory correction.
            # This makes the pre-warm-start path behave like Muon-style momentum/Nesterov.
            update_in = update_pre_ns

        # O_t is the direction after the final Newton-Schulz step.
        O_t, dim_scale = orthogonalize_update_like_muon(self._matrix_view(update_in), steps=group["ns_steps"])

        # W_t = W_{t-1} - lr * (O_t * 0.2 * sqrt(min(rows, cols)) + lambda * W_{t-1})
        p.mul_(1 - group["lr"] * group.get("weight_decay", 0.0))
        p.add_(O_t.reshape_as(p), alpha=-group["lr"] * dim_scale)

        # Update EMA memory with raw projected gradient z_t, not projected momentum/Nesterov direction.
        self._queue_memory_point(z_for_correction, state)
        self._flush_pending_memory(state, param_offset=id(p))

    def _flush_all_pending_memory(self):
        for state in self.state.values():
            if isinstance(state, dict) and "memory_pending_z" in state:
                self._flush_pending_memory(state, force=True)


class MemoryMuon(_MemoryMuonBase):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ema_decay=0.99,
        ema_eps=1e-8,
        step_update=64,
        ema_ws_steps=512,
    ):
        params = list(params)
        self._validate_main_params(params)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self.num_centroids = num_centroids
        self.centroid_dim = centroid_dim
        self.lambda_memory = lambda_memory
        self.memory_init_seed = memory_init_seed
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.step_update = int(step_update)
        self.ema_ws_steps = int(ema_ws_steps)
        if self.step_update < 1:
            raise ValueError("step_update must be >= 1")
        if self.ema_ws_steps < 0:
            raise ValueError("ema_ws_steps must be >= 0")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    state = self.state[p]
                    self._step_muon_param(p, state, group)
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMemoryMuon(_MemoryMuonBase):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ema_decay=0.99,
        ema_eps=1e-8,
        step_update=64,
        ema_ws_steps=512,
    ):
        params = list(params)
        self._validate_main_params(params)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self.num_centroids = num_centroids
        self.centroid_dim = centroid_dim
        self.lambda_memory = lambda_memory
        self.memory_init_seed = memory_init_seed
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.step_update = int(step_update)
        self.ema_ws_steps = int(ema_ws_steps)
        if self.step_update < 1:
            raise ValueError("step_update must be >= 1")
        if self.ema_ws_steps < 0:
            raise ValueError("ema_ws_steps must be >= 0")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                self._step_muon_param(p, state, group)

        return loss


def _make_aux_adam(param_groups):
    aux_groups = []
    for group in param_groups:
        if not group["use_muon"]:
            aux_groups.append({
                "params": group["params"],
                "lr": group.get("lr", 1e-3),
                "betas": group.get("betas", (0.9, 0.999)),
                "eps": group.get("eps", 1e-8),
                "weight_decay": group.get("weight_decay", 0),
            })
    return None if len(aux_groups) == 0 else torch.optim.Adam(aux_groups)

class MemoryMuonWithAuxAdam(_MemoryMuonBase):
    def __init__(
        self,
        param_groups,
        *,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ema_decay=0.99,
        ema_eps=1e-8,
        step_update=64,
        ema_ws_steps=512,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                self._validate_main_params(group["params"])
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)
        self.num_centroids = num_centroids
        self.centroid_dim = centroid_dim
        self.lambda_memory = lambda_memory
        self.memory_init_seed = memory_init_seed
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.step_update = int(step_update)
        self.ema_ws_steps = int(ema_ws_steps)
        if self.step_update < 1:
            raise ValueError("step_update must be >= 1")
        if self.ema_ws_steps < 0:
            raise ValueError("ema_ws_steps must be >= 0")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        state = self.state[p]
                        self._step_muon_param(p, state, group)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])


        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss


class SingleDeviceMemoryMuonWithAuxAdam(_MemoryMuonBase):
    def __init__(
        self,
        param_groups,
        *,
        num_centroids=16,
        centroid_dim=1024,
        lambda_memory=0.01,
        memory_init_seed=None,
        ema_decay=0.99,
        ema_eps=1e-8,
        step_update=64,
        ema_ws_steps=512,
    ):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                self._validate_main_params(group["params"])
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)
        self.num_centroids = num_centroids
        self.centroid_dim = centroid_dim
        self.lambda_memory = lambda_memory
        self.memory_init_seed = memory_init_seed
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.step_update = int(step_update)
        self.ema_ws_steps = int(ema_ws_steps)
        if self.step_update < 1:
            raise ValueError("step_update must be >= 1")
        if self.ema_ws_steps < 0:
            raise ValueError("ema_ws_steps must be >= 0")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    self._step_muon_param(p, state, group)


        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss
