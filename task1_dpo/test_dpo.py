import torch

from dpo import dpo_loss


# ------------------------------------------------------------
# Test 1
# Policy prefers chosen by margin 3.
# Reference has zero preference margin.
# ------------------------------------------------------------

policy_chosen = torch.tensor([4.0])
policy_rejected = torch.tensor([1.0])

ref_chosen = torch.tensor([0.0])
ref_rejected = torch.tensor([0.0])

loss, diagnostics = dpo_loss(
    policy_chosen,
    policy_rejected,
    ref_chosen,
    ref_rejected,
    beta=0.1,
)

print("Test 1")
print("Loss:", loss.item())
print("Logit:", diagnostics["logit_mean"].item())
print(
    "Preference accuracy:",
    diagnostics["preference_accuracy"].item(),
)


# ------------------------------------------------------------
# Test 2
# Policy and reference have exactly the same margin.
#
# Therefore:
#
# policy_margin = 2
# reference_margin = 2
#
# DPO logit = 0
# Loss = -log(sigmoid(0))
#      = log(2)
#      = 0.693147...
# ------------------------------------------------------------

policy_chosen = torch.tensor([3.0])
policy_rejected = torch.tensor([1.0])

ref_chosen = torch.tensor([3.0])
ref_rejected = torch.tensor([1.0])

loss, diagnostics = dpo_loss(
    policy_chosen,
    policy_rejected,
    ref_chosen,
    ref_rejected,
    beta=0.1,
)

expected = torch.log(
    torch.tensor(2.0)
)

print("\nTest 2")
print("Loss:", loss.item())
print(
    "Expected:",
    expected.item(),
)

assert torch.allclose(
    loss,
    expected,
    atol=1e-6,
)

assert abs(
    diagnostics["logit_mean"].item()
) < 1e-6

print(
    "\nDPO objective validation PASSED."
)