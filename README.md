# **Coding Assignment 6: Normalizing Flows and Flow Matching**

</br>

> **Note:** This assignment is **optional** this semester. Point values below are unscaled and are
> retained for reference; see the course Canvas page for how (or whether) it counts toward your grade.

## **Overview**

In CA5 you met two latent-variable generative models — the VAE and the GAN — plus diffusion, which
learns to reverse a fixed noising process. This assignment continues that thread with two further
strategies for learning a map $f$ that turns a boring distribution $z$ (a random normal vector) into
an interesting one $x = f(z)$ imitating the target.

Both notebooks use the same simple "swiss roll" target distribution as CA5.2–5.4, so results are
directly comparable with the VAE, GAN, and diffusion models you already built.

The two methods take opposite approaches to the same problem. **Flow matching** learns a continuous
*velocity field* and generates by integrating an ODE — closely related to diffusion, but with a
simpler, straight-line interpolation path. **RealNVP** instead builds $f$ out of invertible layers
with tractable Jacobian determinants, which buys something neither the VAE, the GAN, nor diffusion
can offer: *exact* log-likelihoods.

**Prerequisite:** Complete **CA5.4 (Diffusion)** first. Several analysis questions here ask you to
compare against the diffusion and GAN models from CA5.

---

## **Notebooks**

### **6.1 — Flow Matching**

Implement the flow matching training objective and a higher-order ODE solver for sampling.

**Topics:**
- Linear interpolation and velocity fields
- Flow matching training objective
- Euler vs. Runge-Kutta integration
- Comparison with diffusion models (CA5.4)

**Point Breakdown (15 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Training Objective** | 4 pts | — |
| **RK4 Solver** | 4 pts | — |
| **Analysis Questions** | — | 7 pts |

---

### **6.2 — RealNVP (Normalizing Flows)**

Implement the affine coupling layer at the heart of RealNVP and understand exact likelihood
computation.

**Topics:**
- Affine coupling transformations (forward and inverse)
- Change of variables formula
- Tractable Jacobian determinants
- Exact vs. approximate likelihood

**Point Breakdown (10 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Coupling Layer — Forward** | 2 pts | — |
| **Coupling Layer — Inverse** | 1 pt | — |
| **Log-Likelihood Loss** | 3 pts | — |
| **Analysis Questions** | — | 4 pts |

---

## **Total Points: 25**

| Notebook | Coding | Analysis | Total |
|----------|--------|----------|-------|
| 6.1 Flow Matching | 8 pts | 7 pts | 15 pts |
| 6.2 RealNVP | 6 pts | 4 pts | 10 pts |
| **Total** | **14 pts** | **11 pts** | **25 pts** |

---

## **Learning Objectives**

By completing this assignment, you will:
1. Implement two further generative modeling paradigms (flow matching, normalizing flows)
2. Understand the relationship between flow matching and diffusion (CA5.4)
3. Understand why invertibility and tractable Jacobians enable exact likelihoods
4. Compare numerical integrators (Euler vs. RK4) and their cost/quality trade-off
5. Analyze computational costs of generation across adversarial, score-based, flow-based, and
   likelihood-based methods

---

## **Data Setup**

### Swiss Roll (both notebooks):
Generated synthetically in each notebook — no external data needed.

### **Installation:**
```bash
pip install torch numpy matplotlib tqdm
```

---

## **Submission Instructions**

Submit **both completed notebooks** by the deadline.

### **Deliverables:**
- `CA6.1_FlowMatching.ipynb`
- `CA6.2_RealNVP.ipynb`

Keep the original filenames. Zip both completed notebooks into a single archive named:

```
CA6-Your-Last-Name.zip
```

For example, a student named Jordan Rivera submits `CA6-Rivera.zip`.

### **Requirements:**
- All TODO sections must be filled in with working code
- All questions answered in markdown cells (marked "YOUR ANSWER:")
- Generated outputs visible (loss plots, sample visualizations, comparisons)
- Code runs without errors from top to bottom

### **Where to submit:**
Post your zip file as a direct **reply** to the **Coding Assignment 6** discussion thread on Canvas.
You'll find that thread in the **Coding Assignments** module for the week this assignment is due.

- Reply to the existing thread — do not start a new one.
- Attach the single zip file — do not upload the notebooks individually.
- If you cannot find the thread, check the course Canvas page or contact the TA (see *Getting Help*).

---

## **Resources**

**Papers Referenced:**
- Flow Matching: Lipman et al. (2023) — Flow Matching for Generative Modeling
- RealNVP: Dinh et al. (2017) — Density Estimation Using Real-valued Non-Volume Preserving
  Transformations

**Additional Reading:**
- [Lilian Weng's Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
- [Lilian Weng's Diffusion Models Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [What are Diffusion Models? (Tutorial)](https://arxiv.org/abs/2208.11970)

---

## **Getting Help**

If you encounter any problems with the notebooks, notice any errors, or have questions about the
setup, please do not hesitate to reach out. You can email me at
**[patel.pranav2@northeastern.edu](mailto:patel.pranav2@northeastern.edu)** or send me a message on
Microsoft Teams.

---

## **Academic Integrity**

- **Collaboration:** Follow course syllabus policy on collaboration
- **Citations:** Credit all external resources, discussions, and AI assistance used
- **Code reuse:** Cite any code adapted from online sources
- **Individual work:** Submit your own implementations — copying code is prohibited

---

**Good luck exploring generative models!**
