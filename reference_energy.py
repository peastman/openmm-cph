from openmm.unit import kilojoules_per_mole, MOLAR_GAS_CONSTANT_R
import numpy as np

class ReferenceEnergyFinder(object):
    def __init__(self, model, pKa, temperature):
        if len(model.titrations) != 1:
            raise ValueError("The model compound must contain a single titratable residue")
        self.model = model
        self.pKa = pKa
        self.temperature = temperature
        self.residueIndex = list(model.titrations.keys())[0]
        self.titration = model.titrations[self.residueIndex]
        if len(self.titration.explicitStates) != 2:
            raise ValueError("Only residues with two states are currently supported")

    def findReferenceEnergies(self, iterations):
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
            self.model.simulation.step(20)
            self.model.attemptMCStep(self.temperature)
        fractions = [[] for _ in range(len(self.model.pH))]
        for i in range(iterations):
            self.model.simulation.step(20)
            self.model.attemptMCStep(self.temperature)
            fractions[self.model.currentPHIndex].append(1.0 if self.titration.protonatedIndex == self.titration.currentIndex else 0.0)
        x = []
        y = []
        for i in range(len(fractions)):
            fraction = np.average(fractions[i])
            if fraction >= 0.01 and fraction <= 0.99:
                x.append(self.model.pH[i])
                y.append(np.log(fraction)-np.log(0.5))

        # Fit a line through the data to better estimate when the fraction is exactly 0.5,
        # and compute the reference energy based on it.

        polynomial = np.polyfit(x, y, 1)
        root = np.roots(polynomial)
        kT = MOLAR_GAS_CONSTANT_R*self.temperature
        self.titration.referenceEnergies[1] += kT*deltaN*np.log(10.0)*(self.pKa-root)
