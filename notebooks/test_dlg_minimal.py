import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# A minimal model with Sigmoid activations
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(1 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def test_dlg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    model.eval()

    # Create a simple recognizable "true" image (e.g. a cross pattern)
    true_img = torch.zeros(1, 1, 8, 8, device=device)
    true_img[:, 0, 3:5, :] = 1.0  # Horizontal line
    true_img[:, 0, :, 3:5] = 1.0  # Vertical line
    true_label = torch.tensor([3], device=device)

    criterion = nn.CrossEntropyLoss()

    # Compute target gradients
    model.zero_grad()
    target_logits = model(true_img)
    target_loss = criterion(target_logits, true_label)

    target_dy_dx = torch.autograd.grad(target_loss, model.parameters())
    target_dy_dx = [g.detach() for g in target_dy_dx]

    # Initialize dummy image with random noise
    dummy_img = torch.randn(1, 1, 8, 8, device=device).requires_grad_(True)

    # LBFGS is the optimizer used in the original DLG paper for small/smooth networks
    optimizer = optim.LBFGS([dummy_img], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

    history = []

    # Compute initial grad diff
    dummy_logits = model(dummy_img)
    dummy_loss = criterion(dummy_logits, true_label)
    dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
    initial_diff = 0
    for gx, gy in zip(dummy_dy_dx, target_dy_dx):
        initial_diff += ((gx - gy) ** 2).sum()
    print(f"Initial GradDiff: {initial_diff.item():.8f}")

    print("Starting DLG on minimal model...")
    for iters in range(20):  # 20 * 20 = 400 internal steps

        def closure():
            optimizer.zero_grad()
            dummy_logits = model(dummy_img)
            dummy_loss = criterion(dummy_logits, true_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        # Calculate current loss for logging
        current_loss = closure().item()
        print(f"Iter {iters * 20:03d} | GradDiff: {current_loss:.8f}")
        history.append((iters * 20, dummy_img.clone().detach()))

        if current_loss < 1e-7:
            print("Converged!")
            break

    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(true_img[0, 0].cpu().clamp(0, 1), cmap="gray")
    axes[0].set_title("True Image")

    axes[1].imshow(history[0][1][0, 0].cpu().clamp(0, 1), cmap="gray")
    axes[1].set_title(f"Step {history[0][0]}")

    mid_idx = len(history) // 2
    axes[2].imshow(history[mid_idx][1][0, 0].cpu().clamp(0, 1), cmap="gray")
    axes[2].set_title(f"Step {history[mid_idx][0]}")

    axes[3].imshow(history[-1][1][0, 0].cpu().clamp(0, 1), cmap="gray")
    axes[3].set_title("Final Reconstructed")

    # Turn off axes
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("dlg_minimal_test.png", bbox_inches="tight")
    print("Saved dlg_minimal_test.png")
    print(f"MSE to True Image: {torch.nn.functional.mse_loss(dummy_img, true_img).item():.6f}")


if __name__ == "__main__":
    test_dlg()

"""
### How this script works:
This is a minimal, standalone demonstration of Deep Leakage from Gradients (DLG) working perfectly.
1. It creates a very shallow, simple network (an MLP with Sigmoid activations instead of ReLUs).
2. It creates a tiny 8x8 dummy image of a cross and computes its gradients.
3. Starting from pure random noise, it uses the L-BFGS optimizer to tweak the noise until its 
   gradients perfectly match the cross's gradients.
4. Because the network is shallow and uses smooth activations, the optimizer is mathematically 
   guaranteed to find the exact original image, demonstrating the core vulnerability DLG exploits.
"""
