import json, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = sys.argv[1] if len(sys.argv) > 1 else "task_2.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

pat = re.compile(r"\[Update (\d+)/(\d+)\] reward=(\S+) \| KL=(\S+) \| policy_loss=(\S+) \| "
                 r"value_loss=(\S+) \| entropy=(\S+) \| clip=(\S+) \| length=(\S+) \| grad=(\S+) \| time=(\S+?)s")

def cell_text(c):
    out = []
    for o in c.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain") or []
        out.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(out)

# standard run = the cell whose output has "PPO STANDARD CONTINUATION" and 20 updates
rows = None
for c in nb["cells"]:
    txt = cell_text(c)
    if "PPO STANDARD CONTINUATION" in txt:
        found = {}
        for m in pat.finditer(txt):          # DEBUG lines repeat update lines; dict dedups
            found[int(m.group(1))] = [float(x) for x in m.groups()[2:]]
        if len(found) == 20:
            rows = [found[k] for k in sorted(found)]
            break
assert rows, "standard run log not found"

cols = ["reward","KL","policy_loss","value_loss","entropy","clip","length","grad","time"]
d = {k: [r[i] for r in rows] for i, k in enumerate(cols)}
x = list(range(1, 21))

panels = [("reward","Reward"),("KL","KL to reference"),("value_loss","Value loss"),
          ("entropy","Entropy"),("length","Response length (tokens)"),("grad","Gradient norm")]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.2), constrained_layout=True)
for ax, (k, t) in zip(axs.ravel(), panels):
    ax.plot(x, d[k], marker="o", ms=3, lw=1.2)
    if k in ("reward","KL"): ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(t, fontsize=10); ax.set_xticks([1,5,10,15,20]); ax.grid(alpha=0.3)
    if ax in axs[1]: ax.set_xlabel("update")
fig.suptitle("Standard PPO continuation (20 updates, one prompt each; clip fraction = 0 throughout)", fontsize=10)
fig.savefig("standard_ppo.png", dpi=200); fig.savefig("standard_ppo.pdf")
print("saved standard_ppo.png/pdf; clip max =", max(d["clip"]))
