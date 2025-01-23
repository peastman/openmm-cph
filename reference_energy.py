from openmm.unit import kilojoules_per_mole, kelvin, is_quantity, MOLAR_GAS_CONSTANT_R
import numpy as np

class ReferenceEnergyFinder(object):
    def __init__(self, model, pKa, temperature):
        """
        Construct a ReferenceEnergyFinder.

        Parameters
        ----------
        model: ConstantPH
            The model for which to determine reference energies.  It must contain a single titratable residue with
            exactly two states.  It does not matter what pH or reference energies were specified when it was created,
            because they will both be overwritten.
        pKa: float
            The experimental pKa of the titratable residue.  Reference energies will be chosen to match it.
        temperature: openmm.unit.Quantity
            The temperature at which the simulation will be run.
        """
        if len(model.titrations) != 1:
            raise ValueError("The model compound must contain a single titratable residue")
        self.model = model
        self.pKa = pKa
        if not is_quantity(temperature):
            temperature = temperature*kelvin
        self.temperature = temperature
        self.residueIndex = list(model.titrations.keys())[0]
        self.titration = model.titrations[self.residueIndex]
        if len(self.titration.explicitStates) != 2:
            raise ValueError("Only residues with two states are currently supported")

    def findReferenceEnergies(self, iterations=20000, substeps=20):
        """
        Compute the reference energies for the states of the model compound.  On exit, they will be stored in
        the ConstantPH object.

        Parameters
        ----------
        iterations: int
            The number of Monte Carlo moves to attempt.  The larger the number, the more tightly converged
            the results will be.
        subsets: int
            The number of dynamics steps to integrate between Monte Carlo moves.
        """
        # Find an initial estimate of the reference energies just by computing the potential
        # energies of the states.

        self.model.setResidueState(self.residueIndex, 0)
        energy0 = self.model.implicitContext.getState(energy=True).getPotentialEnergy()
        self.model.setResidueState(self.residueIndex, 1)
        energy1 = self.model.implicitContext.getState(energy=True).getPotentialEnergy()
        deltaN = self.titration.implicitStates[1].numHydrogens - self.titration.implicitStates[0].numHydrogens
        self.titration.referenceEnergies[0] = 0.0*kilojoules_per_mole
        self.titration.referenceEnergies[1] = energy1-energy0

        # If our initial estimate is exact, the fractions should be equal at pH 0.  Since it probably
        # isn't, simulate it at various pHs to refine the estimate.

        self.model.setPH([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        for i in range(1000):
            self.model.simulation.step(substeps)
            self.model.attemptMCStep(self.temperature)
        fractions = [[] for _ in range(len(self.model.pH))]
        for i in range(iterations):
            self.model.simulation.step(substeps)
            self.model.attemptMCStep(self.temperature)
            fractions[self.model.currentPHIndex].append(1.0 if self.titration.protonatedIndex == self.titration.currentIndex else 0.0)
        x = []
        y = []
        w = []
        for i in range(len(fractions)):
            fraction = np.average(fractions[i])
            if fraction > 0.0 and fraction < 1.0:
                x.append(self.model.pH[i])
                y.append(np.log(fraction)-np.log(0.5))
                if fraction < 0.5:
                    count = sum(fractions[i])
                else:
                    count = len(fractions[i])-sum(fractions[i])
                w.append(np.sqrt(count))

        # Fit a line through the data to better estimate when the fraction is exactly 0.5,
        # and compute the reference energy based on it.

        root = np.roots(np.polyfit(x, y, 1, w=w))[0]
        kT = MOLAR_GAS_CONSTANT_R*self.temperature
        self.titration.referenceEnergies[1] += kT*deltaN*np.log(10.0)*(self.pKa-root)
