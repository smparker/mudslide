# Preparing a Harmonic Model with Turbomole

You can prepare a run to make a harmonic model by optimizing the geometry
and then calculating vibrational frequencies. Here is an example, assuming
you've already set up a control file (through define or otherwise):

```bash
# Step 1: Optimize the geometry
jobex > jobex.out

# Step 2: Calculate vibrational frequencies
aoforce > aoforce.out

# Step 3: Mudslide make harmonic
mudslide-make-harmonic
```
