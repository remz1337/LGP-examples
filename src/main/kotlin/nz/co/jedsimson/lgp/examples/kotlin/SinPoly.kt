package nz.co.jedsimson.lgp.examples.kotlin

import nz.co.jedsimson.lgp.core.environment.DefaultValueProviders
import nz.co.jedsimson.lgp.core.environment.Environment
import nz.co.jedsimson.lgp.core.environment.config.Configuration
import nz.co.jedsimson.lgp.core.environment.config.ConfigurationLoader
import nz.co.jedsimson.lgp.core.environment.constants.GenericConstantLoader
import nz.co.jedsimson.lgp.core.environment.dataset.*
import nz.co.jedsimson.lgp.core.environment.operations.DefaultOperationLoader
import nz.co.jedsimson.lgp.core.evolution.Description
import nz.co.jedsimson.lgp.core.evolution.Problem
import nz.co.jedsimson.lgp.core.evolution.ProblemNotInitialisedException
import nz.co.jedsimson.lgp.core.evolution.Solution
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessContexts
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessFunctions
import nz.co.jedsimson.lgp.core.evolution.model.MasterSlave
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.macro.MacroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.ConstantMutationFunctions
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.MicroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.recombination.linearCrossover.LinearCrossover
import nz.co.jedsimson.lgp.core.evolution.operators.selection.TournamentSelection
import nz.co.jedsimson.lgp.core.evolution.training.DistributedTrainer
import nz.co.jedsimson.lgp.core.evolution.training.TrainingResult
import nz.co.jedsimson.lgp.core.modules.CoreModuleType
import nz.co.jedsimson.lgp.core.modules.ModuleContainer
import nz.co.jedsimson.lgp.core.modules.ModuleInformation
import nz.co.jedsimson.lgp.core.program.Outputs
import nz.co.jedsimson.lgp.lib.base.BaseProgram
import nz.co.jedsimson.lgp.lib.base.BaseProgramOutputResolvers
import nz.co.jedsimson.lgp.lib.base.BaseProgramSimplifier
import nz.co.jedsimson.lgp.lib.generators.EffectiveProgramGenerator
import nz.co.jedsimson.lgp.lib.generators.RandomInstructionGenerator

data class SinPolySolution(
        override val problem: String,
        val result: TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>>
) : Solution<Double>

class SinPolyProblem : Problem<Double, Outputs.Single<Double>, Targets.Single<Double>>() {
    override val name = "SinPoly."

    override val description = Description("f(x) = sin(x) * x + 5\n\trange = Uniform[-5:5]")

    override val configLoader = object : ConfigurationLoader {
        override val information = ModuleInformation("Overrides default configuration for this problem.")

        override fun load(): Configuration {
            val config = Configuration()

            config.initialMinimumProgramLength = 30
            config.initialMaximumProgramLength = 60
            config.minimumProgramLength = 30
            config.maximumProgramLength = 400
            config.operations = listOf(
                    "nz.co.jedsimson.lgp.lib.operations.Addition",
                    "nz.co.jedsimson.lgp.lib.operations.Subtraction",
                    "nz.co.jedsimson.lgp.lib.operations.Multiplication",
                    "nz.co.jedsimson.lgp.lib.operations.Division",
                    "nz.co.jedsimson.lgp.lib.operations.Exponent"
            )
            config.constantsRate = 0.5
            config.constants = listOf(
                    "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"
            )
            config.numCalculationRegisters = 4
            config.populationSize = 1000
            config.generations = 200
            config.numFeatures = 1
            config.microMutationRate = 0.25
            config.macroMutationRate = 0.75
            config.crossoverRate = 0.7

            return config
        }
    }

    private val config = this.configLoader.load()

    override val constantLoader = GenericConstantLoader(
            constants = config.constants,
            parseFunction = String::toDouble
    )

    val datasetLoader = object : DatasetLoader<Double, Targets.Single<Double>> {
        // f(x) = sin(x) * x + 5
        val func = { x: Double -> Math.sin(x) * x + 5.0 }
        val gen = UniformlyDistributedGenerator()

        override val information = ModuleInformation("Generates uniformly distributed samples in the range [-5:5].")

        override fun load(): Dataset<Double, Targets.Single<Double>> {
            val xs = gen.generate(100, -5.0, 5.0).map { v ->
                Sample(
                    listOf(Feature(name = "x", value = v))
                )
            }.toList()

            val ys = xs.map { x ->
                Targets.Single(this.func(x.feature("x").value))
            }

            return Dataset(
                    xs.toList(),
                    ys.toList()
            )
        }
    }

    override val operationLoader = DefaultOperationLoader<Double>(
            operationNames = config.operations
    )

    override val defaultValueProvider = DefaultValueProviders.constantValueProvider(1.0)

    override val fitnessFunctionProvider = {
        FitnessFunctions.SSE
    }

    override val registeredModules = ModuleContainer<Double, Outputs.Single<Double>, Targets.Single<Double>>(
            modules = mutableMapOf(
                    CoreModuleType.InstructionGenerator to { environment ->
                        RandomInstructionGenerator(environment)
                    },
                    CoreModuleType.ProgramGenerator to { environment ->
                        EffectiveProgramGenerator(
                            environment,
                            sentinelTrueValue = 1.0,
                            outputRegisterIndices = listOf(0),
                            outputResolver = BaseProgramOutputResolvers.singleOutput()
                        )
                    },
                    CoreModuleType.SelectionOperator to { environment ->
                        TournamentSelection(environment, tournamentSize = 10, numberOfOffspring = 20)
                    },
                    CoreModuleType.RecombinationOperator to { environment ->
                        LinearCrossover(
                                environment,
                                maximumSegmentLength = 6,
                                maximumCrossoverDistance = 5,
                                maximumSegmentLengthDifference = 3
                        )
                    },
                    CoreModuleType.MacroMutationOperator to { environment ->
                        MacroMutationOperator(
                                environment,
                                insertionRate = 0.67,
                                deletionRate = 0.33
                        )
                    },
                    CoreModuleType.MicroMutationOperator to { environment ->
                        MicroMutationOperator(
                                environment,
                                registerMutationRate = 0.3,
                                operatorMutationRate = 0.4,
                                constantMutationFunc = ConstantMutationFunctions.randomGaussianNoise(
                                    environment.randomState
                                )
                        )
                    },
                    CoreModuleType.FitnessContext to { environment ->
                        FitnessContexts.SingleOutputFitnessContext(environment)
                    }
            )
    )

    override fun initialiseEnvironment() {
        this.environment = Environment(
                this.configLoader,
                this.constantLoader,
                this.operationLoader,
                this.defaultValueProvider,
                this.fitnessFunctionProvider
        )

        this.environment.registerModules(this.registeredModules)
    }

    override fun initialiseModel() {
        this.model = MasterSlave(this.environment)
    }

    override fun solve(): SinPolySolution {
        try {
            val runner = DistributedTrainer(environment, model, runs = 1)
            val result = runner.train(this.datasetLoader.load())

            return SinPolySolution(this.name, result)
        } catch (ex: UninitializedPropertyAccessException) {
            // The initialisation routines haven't been run.
            throw ProblemNotInitialisedException(
                    "The initialisation routines for this problem must be run before it can be solved."
            )
        }
    }
}

class SinPoly {
    companion object Main {
        @JvmStatic fun main(args: Array<String>) {
            // Create a new problem instance, initialise it, and then solve it.
            val problem = SinPolyProblem()
            problem.initialiseEnvironment()
            problem.initialiseModel()
            val solution = problem.solve()
            val simplifier = BaseProgramSimplifier<Double, Outputs.Single<Double>>()

            println("Results:")

            solution.result.evaluations.forEachIndexed { run, res ->
                println("Run ${run + 1} (best fitness = ${res.best.fitness})")
                println(simplifier.simplify(res.best as BaseProgram<Double, Outputs.Single<Double>>))

                println("\nStats (last run only):\n")

                for ((k, v) in res.statistics.last().data) {
                    println("$k = $v")
                }
                println("")
            }

            val avgBestFitness = solution.result.evaluations.map { eval ->
                eval.best.fitness
            }.sum() / solution.result.evaluations.size

            println("Average best fitness: $avgBestFitness")
        }
    }
}
