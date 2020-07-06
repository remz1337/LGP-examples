package nz.co.jedsimson.lgp.examples.kotlin

import kotlinx.coroutines.runBlocking
import nz.co.jedsimson.lgp.core.environment.DefaultValueProviders
import nz.co.jedsimson.lgp.core.environment.Environment
import nz.co.jedsimson.lgp.core.environment.EnvironmentFacade
import nz.co.jedsimson.lgp.core.environment.config.Configuration
import nz.co.jedsimson.lgp.core.environment.config.ConfigurationLoader
import nz.co.jedsimson.lgp.core.environment.constants.GenericConstantLoader
import nz.co.jedsimson.lgp.core.environment.dataset.*
import nz.co.jedsimson.lgp.core.environment.dataset.Target
import nz.co.jedsimson.lgp.core.environment.operations.DefaultOperationLoader
import nz.co.jedsimson.lgp.core.evolution.*
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessFunctions
import nz.co.jedsimson.lgp.core.evolution.model.SteadyState
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.macro.MacroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.ConstantMutationFunctions
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.MicroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.recombination.linearCrossover.LinearCrossover
import nz.co.jedsimson.lgp.core.evolution.operators.selection.BinaryTournamentSelection
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
import nz.co.jedsimson.lgp.core.environment.events.*
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessCase
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessContext
import nz.co.jedsimson.lgp.core.program.Output
import nz.co.jedsimson.lgp.core.program.Program
import java.io.*
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/*
 * An example of setting up an environment to use LGP to find programs for the function `x^2 + 2x + 2`.
 *
 * This example serves as a good way to learn how to use the system and to ensure that everything
 * is working correctly, as some percentage of the time, perfect individuals should be found.
 */

// A solution for this problem consists of the problem's name and a result from
// running the problem with a `Trainer` impl.
data class SR_Bin3Solution(
        override val problem: String,
        val result: TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>>,
        val outputs: List<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>>
) : Solution<Double>

// Define the problem and the necessary components to solve it.
class SR_Bin3Problem(val datasetStream: InputStream) : Problem<Double, Outputs.Single<Double>, Targets.Single<Double>>() {
    override val name = "Symbolic Regression: Binomial 3"

    override val description = Description("f(x) = x^2 + 2x + 1\n\trange = [-10:10:0.5]")

    override val configLoader = object : ConfigurationLoader {
        override val information = ModuleInformation("Overrides default configuration for this problem.")

        override fun load(): Configuration {
            val config = Configuration()

            config.initialMinimumProgramLength = 20
            config.initialMaximumProgramLength = 20
            config.minimumProgramLength = 20
            config.maximumProgramLength = 20
            config.operations = listOf(
                    "nz.co.jedsimson.lgp.lib.operations.Addition",
                    "nz.co.jedsimson.lgp.lib.operations.Subtraction",
                    "nz.co.jedsimson.lgp.lib.operations.Multiplication"
            )
            config.constantsRate = 0.5
            config.constants = listOf("0.0", "1.0", "2.0")
            config.numCalculationRegisters = 4
            config.populationSize = 1000
            config.generations = 1
            config.numFeatures = 1
            config.microMutationRate = 0.4
            config.macroMutationRate = 0.0 // disabled since individuals are fixed length

            return config
        }
    }

    private val config = this.configLoader.load()

    override val constantLoader = GenericConstantLoader(
            constants = config.constants,
            parseFunction = String::toDouble
    )

    /*val datasetLoader = object : DatasetLoader<Double, Targets.Single<Double>> {
        // x^2 + 2x + 1
        val func = { x: Double -> (x * x) + (2 * x) + 1 }
        val gen = SequenceGenerator()

        override val information = ModuleInformation("Generates samples in the range [-10:10:0.5].")

        override fun load(): Dataset<Double, Targets.Single<Double>> {
            val xs = gen.generate(-10.0, 10.0, 0.5, inclusive = true).map { x ->
                Sample(
                        listOf(Feature(name = "x", value = x))
                )
            }

            val ys = xs.map { x ->
                Targets.Single(this.func(x.features[0].value))
            }

            return Dataset(
                    xs.toList(),
                    ys.toList()
            )
        }
    }*/

    val featureIndices = 0 until 1
    val targetIndex = 1
    val datasetLoader = CsvDatasetLoader(
            reader = BufferedReader(
                    // Load from the resource file.
                    InputStreamReader(this.datasetStream)
            ),
            featureParseFunction = { header: Header, row: Row ->
                val features = row.zip(header)
                        .slice(featureIndices)
                        .map { (featureValue, featureName) ->

                            Feature(
                                    name = featureName,
                                    value = featureValue.toDouble()
                            )
                        }

                Sample(features)
            },
            targetParseFunction = { _: Header, row: Row ->
                val target = row[targetIndex]

                Targets.Single(target.toDouble())
            }
    )

    override val operationLoader = DefaultOperationLoader<Double>(
            operationNames = config.operations
    )

    override val defaultValueProvider = DefaultValueProviders.constantValueProvider(1.0)

    override val fitnessFunctionProvider = {
        FitnessFunctions.MSE
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
                        BinaryTournamentSelection(environment, tournamentSize = 2)
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
                                registerMutationRate = 0.5,
                                operatorMutationRate = 0.5,
                                // Use identity func. since the probabilities
                                // of other micro mutations mean that we aren't
                                // modifying constants.
                                constantMutationFunc = ConstantMutationFunctions.identity<Double>()
                        )
                    },
                    CoreModuleType.FitnessContext to { environment ->
                        TracingFitnessContext(environment)
                    }
            )
    )

    override fun initialiseEnvironment() {
        this.environment = Environment(
                this.configLoader,
                this.constantLoader,
                this.operationLoader,
                this.defaultValueProvider,
                this.fitnessFunctionProvider,
                ResultAggregators.BufferedResultAggregator(
                        ResultOutputProviders.CsvResultOutputProvider("results.csv")
                )
        )

        this.environment.registerModules(this.registeredModules)
    }

    override fun initialiseModel() {
        this.model = SteadyState(this.environment)
    }

    override fun solve(): SR_Bin3Solution {
        try {
            /*
            // This is an example of training sequentially in an asynchronous manner.
            val runner = SequentialTrainer(environment, model, runs = 2)

            return runBlocking {
                val job = runner.trainAsync(
                    this@SR_Bin3Problem.datasetLoader.load()
                )

                job.subscribeToUpdates { println("training progress = ${it.progress}%") }

                val result = job.result()

                SR_Bin3Solution(this@SR_Bin3Problem.name, result)
            }
            */

            val traceEvents = mutableListOf<DiagnosticEvent.Trace>()

            EventRegistry.register(object : EventListener<DiagnosticEvent.Trace> {
                override fun handle(event: DiagnosticEvent.Trace) {
                    traceEvents += event
                }
            })

            val fitnessContextEvaluationEvents = mutableListOf<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>>()

            EventRegistry.register(object : EventListener<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>> {
                override fun handle(event: FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>) {
                    fitnessContextEvaluationEvents += event
                }
            })

            //Need to do single runs since the test cases outputs aren't linked to the run (yet)
            val runner = DistributedTrainer(environment, model, runs = 1)

            return runBlocking {
                val job = runner.trainAsync(
                        this@SR_Bin3Problem.datasetLoader.load()
                )

                job.subscribeToUpdates { println("training progress = ${it.progress}") }

                val result = job.result()

                SR_Bin3Solution(this@SR_Bin3Problem.name, result, fitnessContextEvaluationEvents)
            }

        } catch (ex: UninitializedPropertyAccessException) {
            // The initialisation routines haven't been run.
            throw ProblemNotInitialisedException(
                    "The initialisation routines for this problem must be run before it can be solved."
            )
        }
    }
}

class FitnessContextEvaluationEvent<TData, TOutput : Output<TData>>(
        val program: Program<TData, TOutput>,
        val outputs: List<TOutput>
) : Event() {

}

class TracingFitnessContext<TData, TOutput : Output<TData>, TTarget : Target<TData>>(
        environment: EnvironmentFacade<TData, TOutput, TTarget>
) : FitnessContext<TData, TOutput, TTarget>(environment) {

    override val information: ModuleInformation
        get() = TODO()

    private val fitnessFunction by lazy {
        this.environment.fitnessFunctionProvider()
    }

    /**
     * Evaluates the fitness by performing the following steps:
     *
     * 1. Finds the effective program
     * 2. Writes each fitness case to the programs register set
     * 3. Executes the program and collects the output from each fitness case
     * 4. Executes the fitness function from the given environment
     */
    override fun fitness(program: Program<TData, TOutput>, fitnessCases: List<FitnessCase<TData, TTarget>>): Double {
        // Make sure the programs effective instructions have been found
        program.findEffectiveProgram()

        // Collect the results of the program for each fitness case.
        val outputs = fitnessCases.map { case ->
            // Make sure the registers are in a default state
            program.registers.reset()

            // Load the case
            program.registers.writeInstance(case.features)

            // Run the program...
            program.execute()

            // ... and gather a result from the program.
            program.output()
        }

        // Create an event with the program and its outputs
        EventDispatcher.dispatch(FitnessContextEvaluationEvent(program, outputs))

        // Copy the fitness to the program for later accesses
        program.fitness = fitnessFunction(outputs, fitnessCases)

        return program.fitness
    }

}

class SR_Bin3 {
    companion object Main {

        private val datasetStream = this::class.java.classLoader.getResourceAsStream("datasets/SR_Bin3.csv")

        @JvmStatic fun main(args: Array<String>) {
            System.setProperty("LGP.LogLevel", "debug")

            // Create a new problem instance, initialise it, and then solve it.
            val problem = SR_Bin3Problem(datasetStream)
            problem.initialiseEnvironment()
            problem.initialiseModel()
            val solution = problem.solve()
            val simplifier = BaseProgramSimplifier<Double, Outputs.Single<Double>>()

            println("Exporting outputs to CSV...")

            var CSV_HEADER = "program"
            for(output_it in solution.outputs[0].outputs.indices){
                CSV_HEADER += ",testcase${output_it}"
            }

            var fileWriter: FileWriter? = null

            try {
                var now = DateTimeFormatter
                        .ofPattern("yyyy-MM-dd_HH-mm-ss")
                        .withZone(ZoneOffset.systemDefault())
                        .format(Instant.now())

                fileWriter = FileWriter("outputs_${now}.csv")

                fileWriter.append(CSV_HEADER)
                fileWriter.append('\n')

                for (program in solution.outputs) {
                    fileWriter.append('"'+program.program.instructions.toString()+'"')
                    for (testcase in program.outputs){
                        fileWriter.append(',')
                        fileWriter.append(testcase.value.toString())
                    }
                    fileWriter.append('\n')
                }

                println("Write CSV successfully!")
            } catch (e: Exception) {
                println("Writing CSV error!")
                e.printStackTrace()
            } finally {
                try {
                    fileWriter!!.flush()
                    fileWriter.close()
                } catch (e: IOException) {
                    println("Flushing/closing error!")
                    e.printStackTrace()
                }
            }

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
