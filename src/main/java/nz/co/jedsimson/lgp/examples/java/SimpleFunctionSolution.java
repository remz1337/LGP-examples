package nz.co.jedsimson.lgp.examples.java;

import nz.co.jedsimson.lgp.core.environment.dataset.Targets;
import nz.co.jedsimson.lgp.core.evolution.Solution;
import nz.co.jedsimson.lgp.core.evolution.training.TrainingResult;
import nz.co.jedsimson.lgp.core.program.Outputs;
import org.jetbrains.annotations.NotNull;

/**
 * A re-implementation of {@link SimpleFunctionSolution} to showcase Java interoperability.
 */
public class SimpleFunctionSolution implements Solution<Double> {

    private String problem;
    private TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>> result;

    SimpleFunctionSolution(String problem, TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>> result) {
        this.problem = problem;
        this.result = result;
    }

    @NotNull
    @Override
    public String getProblem() {
        return this.problem;
    }

    public TrainingResult<Double, Outputs.Single<Double>, Targets.Single<Double>> getResult() {
        return this.result;
    }
}
