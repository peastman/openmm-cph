from openmm import Context, NonbondedForce, GBSAOBCForce
from openmm.app import element, Modeller, Simulation
from openmm.unit import nanometers, kilojoules_per_mole, kelvin, is_quantity, MOLAR_GAS_CONSTANT_R
from copy import deepcopy
import numpy as np

class ResidueState(object):
    def __init__(self, residueIndex, atomIndices, particleParameters, exceptionParameters, numHydrogens):
        self.residueIndex = residueIndex
        self.atomIndices = atomIndices
        self.particleParameters = particleParameters
        self.exceptionParameters = exceptionParameters
        self.numHydrogens = numHydrogens

class ResidueTitration(object):
    def __init__(self, variants, referenceEnergies):
        self.variants = variants
        self.referenceEnergies = referenceEnergies
        self.explicitStates = []
        self.implicitStates = []
        self.protonatedIndex = -1
        self.currentIndex = -1

class ConstantPh(object):
    def __init__(self, topology, positions, explicitForceField, implicitForceField, residueVariants, referenceEnergies, relaxationSteps, explicitArgs, implicitArgs, integrator, relaxationIntegrator, platform=None, properties=None):
        self._explicitArgs = explicitArgs
        self._implicitArgs = implicitArgs
        self.relaxationSteps = relaxationSteps
        self.titrations = {}
        for resIndex, variants in residueVariants.items():
            energies = []
            for e in referenceEnergies[resIndex]:
                # if is_quantity(e):
                #     e = e.value_in_unit(kilojoules_per_mole)
                energies.append(e)
            self.titrations[resIndex] = ResidueTitration(variants, energies)
        implicitToExplicitResidueMap = []
        solventResidues = []

        # Build the implicit solvent topology by removing water and ions.

        ionElements = (element.cesium, element.potassium, element.lithium, element.sodium, element.rubidium,
                       element.chlorine, element.bromine, element.fluorine, element.iodine)
        for residue in topology.residues():
            if residue.name == 'HOH' or (len(residue) == 1 and next(residue.atoms()).element in ionElements):
                solventResidues.append(residue)
            else:
                implicitToExplicitResidueMap.append(residue.index-len(solventResidues))
        modeller = Modeller(topology, positions)
        modeller.delete(solventResidues)
        implicitTopology = modeller.topology
        implicitPositions = modeller.positions

        # Loop over variants to construct a ResidueState for every variant of every titratable residue.

        variantIndex = 0
        finished = False
        explicitVariants = [None]*topology.getNumResidues()
        implicitVariants = [None]*implicitTopology.getNumResidues()
        while not finished:
            finished = True

            # Build the explicit solvent states.

            for resIndex, variants in residueVariants.items():
                if variantIndex < len(variants):
                    finished = False
                    explicitVariants[resIndex] = variants[variantIndex]
            explicitStates = self._findResidueStates(topology, positions, explicitForceField, explicitVariants, explicitArgs)

            # Build the implicit solvent states.

            for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
                if explicitIndex in residueVariants:
                    variants = residueVariants[explicitIndex]
                    if variantIndex < len(variants):
                        implicitVariants[implicitIndex] = variants[variantIndex]
            implicitStates = self._findResidueStates(implicitTopology, implicitPositions, implicitForceField, implicitVariants, implicitArgs)
            assert len(explicitStates) == len(implicitStates)

            # Add them to the ResidueTitration.

            for explicitState, implicitState in zip(explicitStates, implicitStates):
                titration = self.titrations[explicitState.residueIndex]
                if variantIndex < len(titration.variants):
                    titration.explicitStates.append(explicitState)
                    titration.implicitStates.append(implicitState)
            variantIndex += 1

        # Create final versions of the topologies, including the fully protonated versions of all residues.

        for titration in self.titrations.values():
            titration.protonatedIndex = np.argmax([len(state.atomIndices) for state in titration.explicitStates])
        variants = [None]*topology.getNumResidues()
        for resIndex in residueVariants:
            titration = self.titrations[resIndex]
            variants[resIndex] = titration.variants[titration.protonatedIndex]
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=explicitForceField, variants=variants)
        self.explicitTopology = modeller.topology
        explicitPositions = modeller.positions
        variants = [None]*implicitTopology.getNumResidues()
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            if explicitIndex in residueVariants:
                titration = self.titrations[explicitIndex]
                variants[explicitIndex] = titration.variants[titration.protonatedIndex]
        modeller = Modeller(implicitTopology, implicitPositions)
        modeller.addHydrogens(forcefield=implicitForceField, variants=variants)
        self.implicitTopology = modeller.topology
        implicitPositions = modeller.positions

        # Create systems for them.  Also create a third system that is identical to the explicit one,
        # but freezes non-solvent atoms.

        explicitSystem = explicitForceField.createSystem(self.explicitTopology, **explicitArgs)
        implicitSystem = implicitForceField.createSystem(self.implicitTopology, **implicitArgs)
        relaxationSystem = deepcopy(explicitSystem)
        for residue in self.explicitTopology.residues():
            if residue.name != 'HOH' and (len(residue) > 1 or next(residue.atoms()).element not in ionElements):
                for atom in residue.atoms():
                    relaxationSystem.setParticleMass(atom.index, 0.0)

        # For each ResidueTitration, identify the fully protonated state.  Replace the other states
        # with ones that include all protons, setting the parameters of the missing ones to 0.

        for titration in self.titrations.values():
            protonated = titration.protonatedIndex
            titration.currentIndex = protonated
            explicitProtonatedParams = titration.explicitStates[protonated].particleParameters
            implicitProtonatedParams = titration.implicitStates[protonated].particleParameters
            explicitProtonatedExceptionParams = titration.explicitStates[protonated].exceptionParameters
            implicitProtonatedExceptionParams = titration.implicitStates[protonated].exceptionParameters
            for i in range(len(titration.explicitStates)):
                if i != protonated:
                    oldExplicit = titration.explicitStates[i]
                    oldImplicit = titration.implicitStates[i]
                    newExplicit = deepcopy(titration.explicitStates[protonated])
                    newImplicit = deepcopy(titration.implicitStates[protonated])
                    newExplicit.numHydrogens = oldExplicit.numHydrogens
                    newImplicit.numHydrogens = oldImplicit.numHydrogens
                    for forceIndex in newExplicit.particleParameters:
                        params = oldExplicit.particleParameters[forceIndex]
                        for atomName in newExplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newExplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newExplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(explicitProtonatedParams[forceIndex][atomName], explicitSystem.getForce(forceIndex))
                    for forceIndex in newExplicit.exceptionParameters:
                        params = oldExplicit.exceptionParameters[forceIndex]
                        for key in newExplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newExplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newExplicit.exceptionParameters[forceIndex][key] = [0.0]+list(explicitProtonatedExceptionParams[forceIndex][key][1:])
                    for forceIndex in newImplicit.particleParameters:
                        params = oldImplicit.particleParameters[forceIndex]
                        for atomName in newImplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newImplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newImplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(implicitProtonatedParams[forceIndex][atomName], implicitSystem.getForce(forceIndex))
                    for forceIndex in newImplicit.exceptionParameters:
                        params = oldImplicit.exceptionParameters[forceIndex]
                        for key in newImplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newImplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newImplicit.exceptionParameters[forceIndex][key] = [0.0]+list(implicitProtonatedExceptionParams[forceIndex][key][1:])
                    titration.explicitStates[i] = newExplicit
                    titration.implicitStates[i] = newImplicit

        # Create contexts or simulations for all the systems.

        self.simulation = Simulation(self.explicitTopology, explicitSystem, integrator, platform, properties)
        platform = self.simulation.context.getPlatform()
        if properties is None:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform)
            self.relaxationContext = Context(relaxationSystem, relaxationIntegrator, platform)
        else:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform, properties)
            self.relaxationContext = Context(relaxationSystem, relaxationIntegrator, platform, properties)
        self.simulation.context.setPositions(explicitPositions)
        self.relaxationContext.setPositions(explicitPositions)
        self.implicitContext.setPositions(implicitPositions)

        # Record the mapping from implicit system atoms to explicit system atoms.  We need this
        # for copying positions.

        implicitAtomIndex = [None]*implicitSystem.getNumParticles()
        explicitResidues = list(self.explicitTopology.residues())
        implicitResidues = list(self.implicitTopology.residues())
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            explicitRes = explicitResidues[explicitIndex]
            implicitRes = implicitResidues[implicitIndex]
            explicitAtoms = {atom.name: atom.index for atom in explicitRes.atoms()}
            for atom in implicitRes.atoms():
                implicitAtomIndex[atom.index] = explicitAtoms[atom.name]
        self.implicitAtomIndex = np.array(implicitAtomIndex)

        # Record the indices of nonbonded exceptions in each of the contexts.

        self.explicitExceptionIndex = self._findExceptionIndices(explicitSystem, self.explicitTopology)
        self.implicitExceptionIndex = self._findExceptionIndices(implicitSystem, self.implicitTopology)

    def attemptMCStep(self, temperature, pH):
        # Copy the positions to the implicit context.

        explicitPositions = self.simulation.context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(nanometers)
        implicitPositions = explicitPositions[self.implicitAtomIndex]
        self.implicitContext.setPositions(implicitPositions)

        # Process the residues in random order.

        anyChange = False
        for resIndex in np.random.permutation(list(self.titrations)):
            titration = self.titrations[resIndex]
            numStates = len(titration.implicitStates)

            # Select a new state for it.

            if numStates == 2:
                stateIndex = 1-titration.currentIndex
            else:
                stateIndex = titration.currentIndex
                while stateIndex == titration.currentIndex:
                    stateIndex = np.random.randint(numStates)

            # Compute the energy of the implicit solvent system in the current and new states.

            currentEnergy = self.implicitContext.getState(energy=True).getPotentialEnergy()
            self._applyStateToContext(titration.implicitStates[stateIndex], self.implicitContext, self.implicitExceptionIndex)
            newEnergy = self.implicitContext.getState(energy=True).getPotentialEnergy()

            # Decide whether to accept the new state.

            if not is_quantity(temperature):
                temperature = temperature*kelvin
            kT = (MOLAR_GAS_CONSTANT_R*temperature)
            deltaRefEnergy = (titration.referenceEnergies[stateIndex] - titration.referenceEnergies[titration.currentIndex])
            deltaN = titration.implicitStates[stateIndex].numHydrogens - titration.implicitStates[titration.currentIndex].numHydrogens
            w = (newEnergy-currentEnergy-deltaRefEnergy)/kT + deltaN*np.log(10.0)*pH
            if w > 0.0 and np.exp(-w) < np.random.random():
                # Restore the previous state.

                self._applyStateToContext(titration.implicitStates[titration.currentIndex], self.implicitContext, self.implicitExceptionIndex)
                continue
            anyChange = True

            # Apply the new state.

            titration.currentIndex = stateIndex
            self.relaxationContext.setPositions(explicitPositions)
            self._applyStateToContext(titration.explicitStates[stateIndex], self.simulation.context, self.explicitExceptionIndex)
            self._applyStateToContext(titration.explicitStates[stateIndex], self.relaxationContext, self.explicitExceptionIndex)

        # If anything changed, run some dynamics to let the water relax.

        if anyChange:
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            relaxedPositions = self.relaxationContext.getState(positions=True).getPositions(asNumpy=True)
            self.simulation.context.setPositions(relaxedPositions)

    def setResidueState(self, residueIndex, stateIndex, relax=False):
        titration = self.titrations[residueIndex]
        self._applyStateToContext(titration.explicitStates[stateIndex], self.simulation.context, self.explicitExceptionIndex)
        self._applyStateToContext(titration.explicitStates[stateIndex], self.relaxationContext, self.explicitExceptionIndex)
        self._applyStateToContext(titration.implicitStates[stateIndex], self.implicitContext, self.implicitExceptionIndex)
        titration.currentIndex = stateIndex
        if relax:
            self.relaxationContext.setPositions(self.simulation.context.getState(positions=True).getPositions(asNumpy=True))
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            self.simulation.context.setPositions(self.relaxationContext.getState(positions=True).getPositions(asNumpy=True))

    def _findResidueStates(self, topology, positions, forcefield, variants, ffargs):
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=forcefield, variants=variants)
        system = forcefield.createSystem(modeller.topology, **ffargs)
        atoms = list(modeller.topology.atoms())
        residues = list(modeller.topology.residues())
        states = []
        for residue, variant in zip(residues, variants):
            if variant is not None:
                atomIndices = {atom.name: atom.index for atom in residue.atoms()}
                particleParameters = {}
                exceptionParameters = {}
                for i, force in enumerate(system.getForces()):
                    try:
                        particleParameters[i] = {atom.name: force.getParticleParameters(atom.index) for atom in residue.atoms()}
                    except:
                        pass
                    if isinstance(force, NonbondedForce):
                        exceptionParameters[i] = {}
                        for j in range(force.getNumExceptions()):
                            p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(j)
                            atom1 = atoms[p1]
                            atom2 = atoms[p2]
                            if atom1.residue == residue and atom2.residue == residue:
                                exceptionParameters[i][(residue.index, atom1.name, atom2.name)] = (chargeProd, sigma, epsilon)
                numHydrogens = sum(1 for atom in residue.atoms() if atom.element == element.hydrogen)
                states.append(ResidueState(residue.index, atomIndices, particleParameters, exceptionParameters, numHydrogens))
        return states

    def _findExceptionIndices(self, system, topology):
        indices = {}
        atoms = list(topology.atoms())
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue == atom2.residue:
                        indices[(atom1.residue.index, atom1.name, atom2.name)] = i
                        indices[(atom1.residue.index, atom2.name, atom1.name)] = i
        return indices

    def _get_zero_parameters(self, original_parameters, force):
        p = list(original_parameters)
        if isinstance(force, NonbondedForce) or isinstance(force, GBSAOBCForce):
            p[0] = 0.0
        else:
            for i in range(force.getNumPerParticleParameters()):
                if force.getPerParticleParameterName(i) == 'charge':
                    p[i] = 0.0
        return tuple(p)

    def _applyStateToContext(self, state, context, exceptionIndex):
        for forceIndex, params in state.particleParameters.items():
            force = context.getSystem().getForce(forceIndex)
            for atomName, atomParams in params.items():
                atomIndex = state.atomIndices[atomName]
                try:
                    # Custom forces take the parameters as a single tuple.
                    force.setParticleParameters(atomIndex, atomParams)
                except:
                    # Standard forces take them as separate arguments.
                    force.setParticleParameters(atomIndex, *atomParams)
            if isinstance(force, NonbondedForce):
                for key, exceptionParams in state.exceptionParameters[forceIndex].items():
                    p = force.getExceptionParameters(exceptionIndex[key])
                    force.setExceptionParameters(exceptionIndex[key], p[0], p[1], *exceptionParams)
            force.updateParametersInContext(context)