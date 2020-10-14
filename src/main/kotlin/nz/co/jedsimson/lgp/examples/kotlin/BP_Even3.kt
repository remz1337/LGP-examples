package nz.co.jedsimson.lgp.examples.kotlin

import nz.co.jedsimson.lgp.core.environment.DefaultValueProviders
import nz.co.jedsimson.lgp.core.environment.Environment
import nz.co.jedsimson.lgp.core.environment.config.Configuration
import nz.co.jedsimson.lgp.core.environment.config.ConfigurationLoader
import nz.co.jedsimson.lgp.core.environment.constants.DoubleConstantLoader
import nz.co.jedsimson.lgp.core.environment.dataset.*
import nz.co.jedsimson.lgp.core.environment.events.EventListener
import nz.co.jedsimson.lgp.core.environment.events.EventRegistry
import nz.co.jedsimson.lgp.core.environment.operations.DefaultOperationLoader
import nz.co.jedsimson.lgp.core.evolution.*
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessCase
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessContexts
import nz.co.jedsimson.lgp.core.evolution.fitness.FitnessFunction
import nz.co.jedsimson.lgp.core.evolution.fitness.SingleOutputFitnessFunction
import nz.co.jedsimson.lgp.core.evolution.model.SteadyState
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.macro.MacroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.ConstantMutationFunctions
import nz.co.jedsimson.lgp.core.evolution.operators.mutation.micro.MicroMutationOperator
import nz.co.jedsimson.lgp.core.evolution.operators.recombination.linearCrossover.LinearCrossover
import nz.co.jedsimson.lgp.core.evolution.operators.selection.TournamentSelection
import nz.co.jedsimson.lgp.core.evolution.training.SequentialTrainer
import nz.co.jedsimson.lgp.core.evolution.training.TrainingResult
import nz.co.jedsimson.lgp.core.modules.CoreModuleType
import nz.co.jedsimson.lgp.core.modules.ModuleContainer
import nz.co.jedsimson.lgp.core.modules.ModuleInformation
import nz.co.jedsimson.lgp.core.program.Outputs
import nz.co.jedsimson.lgp.core.program.instructions.BinaryOperation
import nz.co.jedsimson.lgp.core.program.registers.Argument
import nz.co.jedsimson.lgp.core.program.registers.Arguments
import nz.co.jedsimson.lgp.core.program.registers.RegisterIndex
import nz.co.jedsimson.lgp.lib.base.BaseProgram
import nz.co.jedsimson.lgp.lib.base.BaseProgramOutputResolvers
import nz.co.jedsimson.lgp.lib.base.BaseProgramSimplifier
import nz.co.jedsimson.lgp.lib.base.BaseProgramTranslator
import nz.co.jedsimson.lgp.lib.generators.EffectiveProgramGenerator
import nz.co.jedsimson.lgp.lib.generators.RandomInstructionGenerator
import nz.co.jedsimson.lgp.lib.operations.toBoolean
import nz.co.jedsimson.lgp.lib.operations.toDouble
import java.io.*
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

private val match: SingleOutputFitnessFunction<Double> = object : SingleOutputFitnessFunction<Double>() {

    override fun fitness(outputs: List<Outputs.Single<Double>>, cases: List<FitnessCase<Double, Targets.Single<Double>>>): Double {
        val mismatches = cases.zip(outputs).filter { (case, actual) ->
            val expected = case.target.value

            actual.value != expected
        }.count()

        return mismatches.toDouble()
    }
}


/**
 * Defines what a solution for this problem looks like.
 */
class BP_Even3ExperimentSolution(
    override val problem: String,
    val result: TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>>,
    val outputs: List<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>>,
    val dataset: Dataset<Double, Targets.Single<Double>>
) : Solution<Double>

/**
 * Defines the problem.
 */
class BP_Even3Experiment(
    val datasetStream: InputStream
) : Problem<Double, Outputs.Single<Double>, Targets.Single<Double>>() {

    // 1. Give the problem a name and simple description.
    override val name = "Boolean Parity Even 3"
    override val description = Description(
        "This program must return true if the majority of inputs are true, otherwise false"
    )

    // 2. Define the necessary dependencies to build a problem.
    override val configLoader = object : ConfigurationLoader {
        override val information = ModuleInformation("Overrides default configuration for this problem.")

        override fun load(): Configuration {
            val config = Configuration()

            config.initialMinimumProgramLength = 20
            config.initialMaximumProgramLength = 20
            config.minimumProgramLength = 20
            config.maximumProgramLength = 20
            config.operations = listOf(
                "nz.co.jedsimson.lgp.lib.operations.Nand",
                "nz.co.jedsimson.lgp.lib.operations.Nor"
            )
            config.constantsRate = 0.0
            config.numCalculationRegisters = 5
            config.populationSize = 10000
            config.generations = 1
            config.numFeatures = 3
            config.microMutationRate = 0.8
            config.macroMutationRate = 0.0//disabled
            config.crossoverRate = 0.0 //disabled
            config.branchInitialisationRate = 0.0

            return config
        }
    }
    // To prevent reloading configuration in this module.
    private val configuration = this.configLoader.load()
    // Load constants from the configuration as double values.
    override val constantLoader = DoubleConstantLoader(constants = this.configuration.constants)
    // Load operations from the configuration (operations are resolved using their class name).
    override val operationLoader = DefaultOperationLoader<Double>(operationNames = this.configuration.operations)
    // Set the default value of any registers to 1.0.
    override val defaultValueProvider = DefaultValueProviders.constantValueProvider(1.0)
    // Use the mean-squared error fitness function.
    override val fitnessFunctionProvider = { match }
    // Define the modules to be used for the core evolutionary operations.
    override val registeredModules = ModuleContainer<Double, Outputs.Single<Double>, Targets.Single<Double>>(
        modules = mutableMapOf(
            // Generate instructions using the built-in instruction generator.
            CoreModuleType.InstructionGenerator to { environment ->
                RandomInstructionGenerator(environment)
            },
            // Generate programs using the built-in programs generator.
            CoreModuleType.ProgramGenerator to { environment ->
                EffectiveProgramGenerator(
                    environment,
                    sentinelTrueValue = 0.0, // Determines the value that represents a boolean "true".
                    outputRegisterIndices = listOf(6, 7), // Two program outputs
                    outputResolver = BaseProgramOutputResolvers.singleOutput()
                )
            },
            // Perform selection using the built-in tournament selection.
            CoreModuleType.SelectionOperator to { environment ->
                TournamentSelection(environment, tournamentSize = 2, numberOfOffspring = 200)
            },
            // Combine individuals using the linear crossover operator.
            CoreModuleType.RecombinationOperator to { environment ->
                LinearCrossover(
                    environment,
                    maximumSegmentLength = 6,
                    maximumCrossoverDistance = 5,
                    maximumSegmentLengthDifference = 3
                )
            },
            // Use the built-in macro-mutation operator.
            CoreModuleType.MacroMutationOperator to { environment ->
                MacroMutationOperator(
                    environment,
                    insertionRate = 0.67,
                    deletionRate = 0.33
                )
            },
            // Use the built-in micro-mutation operator.
            CoreModuleType.MicroMutationOperator to { environment ->
                MicroMutationOperator(
                    environment,
                    registerMutationRate = 0.5,
                    operatorMutationRate = 0.5,
                    constantMutationFunc = ConstantMutationFunctions.randomGaussianNoise(this.environment.randomState)
                )
            },
            // Use the Single-output fitness context -- meaning that program fitness will be evaluated
            // using Single program outputs and the fitness function specified earlier in this definition.
            CoreModuleType.FitnessContext to { environment ->
                //FitnessContexts.SingleOutputFitnessContext(environment)
                TracingFitnessContext(environment)
            }
        )
    )

    // 3. Describe how to initialise the problem's environment.
    override fun initialiseEnvironment() {
        this.environment = Environment(
            this.configLoader,
            this.constantLoader,
            this.operationLoader,
            this.defaultValueProvider,
            this.fitnessFunctionProvider,
            // Collect results and output them to the file "result.csv".
            ResultAggregators.BufferedResultAggregator(
                ResultOutputProviders.CsvResultOutputProvider(
                    filename = "results.csv"
                )
            )
        )

        this.environment.registerModules(this.registeredModules)
    }

    // 4. Specify which evolution model should be used to solve the problem.
    override fun initialiseModel() {
        this.model = SteadyState(this.environment)
    }

    // 5. Describe the steps required to solve the problem using the definition given above.
    override fun solve(): BP_Even3ExperimentSolution {
        // Indices of the feature variables
        // a, b, c_in
        val featureIndices = 0 until 3
        // Indices of the target variables.
        // c_out, s
        val targetIndex = 3

        // Load the data set
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
                //val targets = row.slice(targetIndices).map { target -> target.toDouble() }

                //Targets.Single(targets)

                val target = row[targetIndex]

                Targets.Single(target.toDouble())
            }
        )


        val fitnessContextEvaluationEvents = mutableListOf<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>>()

        EventRegistry.register(object : EventListener<FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>> {
            override fun handle(event: FitnessContextEvaluationEvent<Double, Outputs.Single<Double>>) {
                fitnessContextEvaluationEvents += event
            }
        })


        val dataset = datasetLoader.load()

        // Print details about the data set and configuration before beginning the evolutionary process.
        println("\nDataset details:")
        println("numFeatures = ${dataset.features}, numSamples = ${dataset.samples}")
        println()
        println(this.configuration)

        // Train using the built-in sequential trainer.
        val trainer = SequentialTrainer(
            this.environment,
            this.model,
            runs = this.configuration.numberOfRuns
        )

        // Return a solution the problem.
        return BP_Even3ExperimentSolution(
            problem = this.name,
            result = trainer.train(dataset),
            outputs = fitnessContextEvaluationEvents,
            dataset = dataset
        )
    }
}

class BP_Even3 {
    companion object Main {

        private val datasetStream = this::class.java.classLoader.getResourceAsStream("datasets/BP_Even3.csv")

        @JvmStatic fun main(args: Array<String>) {
            val problem = BP_Even3Experiment(datasetStream)
            problem.initialiseEnvironment()
            problem.initialiseModel()

            val solution = problem.solve()
            val simplifier = BaseProgramSimplifier<Double, Outputs.Single<Double>>()
            val programTranslator = BaseProgramTranslator<Double, Outputs.Single<Double>>(includeMainFunction = false)

            println("Exporting outputs to CSV...")

            var CSV_HEADER = "program,fitness"
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
                    fileWriter.append(program.program.instructions.toString().replace(",","").replace("[","").replace("]",""))
                    fileWriter.append(','+program.program.fitness.toString())
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

                println(programTranslator.translate(res.best))
            }
        }
    }
}
