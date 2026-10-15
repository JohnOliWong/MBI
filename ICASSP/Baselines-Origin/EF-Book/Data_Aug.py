import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)

def z_score(signal, eps=1e-6):
	mean = signal.mean(dim=(0,2), keepdim=True)
	std = signal.std(dim=(0,2), keepdim=True)
	std = torch.max(std, torch.tensor(eps, device=signal.device))
	signal = (signal - mean) / std
	return signal

def time_shift(signal, shift_ratio=0.2):
	shift = int(signal.shape[2] * shift_ratio * (torch.rand(1) - 0.5))
	return torch.roll(signal, shift, dims=2)

def noise(signal, snr_db=20):
	signal_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)
	noise_power = signal_power / (10 ** (snr_db / 10))
	noise = torch.randn_like(signal) * torch.sqrt(noise_power)
	return signal + noise

def amp(signal, scale=(0.8, 1.2), prob=0.5):
	mask = (torch.rand(signal.shape[0]) < prob).view(-1, 1, 1)
	ratio = scale[0] + torch.rand(signal.shape[0], 1, 1) * (scale[1] - scale[0])
	return signal * (ratio * mask + ~mask * 1.0)

def channel_shuffle(signal, p_swap=0.2):
	trial, channel, time = signal.shape
	for i in range(trial):
		for c in range(channel):
			if torch.rand(1) < p_swap:
				swap_idx = torch.randint(0, channel, (1,)).item()
				signal[i, [c, swap_idx]] = signal[i, [swap_idx, c]]
	return signal

def data_augmentation(signal):
	signal = z_score(signal)
	signal = time_shift(signal)
	signal = noise(signal)
	signal = amp(signal)
	signal = channel_shuffle(signal)
	return signal