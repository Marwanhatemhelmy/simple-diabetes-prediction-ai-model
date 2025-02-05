import java.util.HashMap;

class OutputLayer{
    public OutputNeuron[] outputNeurons;
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> firstMomentumW = new HashMap<>();
    // {0:{0:{0:0.0}}}
    // t  ni  wi wv
    // t = time step
    // ni = neuron index
    // wi = weight index
    // wv = weight value
    // example:
    // {1:{0:{0:0.4,1:0.7}}}
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> secondMomentumW = new HashMap<>();
    public HashMap<Integer, HashMap<Integer, Double>> firstMomentumB = new HashMap<>();
    public HashMap<Integer, HashMap<Integer, Double>> secondMomentumB = new HashMap<>();
    public int timeStep=1;
    public int batchSize = 1;
    private String lossFunction;
    public OutputLayer(){}
    public OutputLayer(OutputNeuron[] outputNeurons){
        this.outputNeurons = outputNeurons;
    }

    // all the loss functions are also avalible at the output layer lever
    public double meanSquaredError(){
        if (this.outputNeurons == null){return 0;}
        double errorSum = 0;
        for (int i=0;i<=outputNeurons.length-1;i++){
            errorSum += Math.pow(this.outputNeurons[i].trueValue - outputNeurons[i].sigmoidActivationFunction(), 2);
        }
        return (1.0/(double)this.outputNeurons.length)*errorSum;
    }

    public double binaryCrossEntropy(){
        if (this.outputNeurons == null){return 0;}
        double errorSum = 0;
        for (int i=0;i<=outputNeurons.length-1;i++){
            errorSum += (this.outputNeurons[i].trueValue*Math.log(this.outputNeurons[i].sigmoidActivationFunction())+(1-this.outputNeurons[i].trueValue)*Math.log(1-this.outputNeurons[i].sigmoidActivationFunction()));
        }
        errorSum *= -(1.0/(double)this.batchSize);
        return errorSum;
    }

    public double binaryCrossEntropyWithLogits(){
        if (this.outputNeurons == null){return 0;}
        double errorSum = 0;
        for (int i=0;i<=outputNeurons.length-1;i++){
            errorSum += (this.outputNeurons[i].trueValue*Math.log(this.outputNeurons[i].sigmoidActivationFunction())+(1-this.outputNeurons[i].trueValue)*Math.log(1-this.outputNeurons[i].sigmoidActivationFunction()));
        }
        errorSum *= -(1.0/(double)this.batchSize);
        return errorSum;
    }

    // this loss derivatives hashmap is a hashmap with the values of the derivatives
    // of the different loss functions , and is needed to get updated after each row gradient update
    public void updateOutputNeuronsLossDerivativesHashMap(){
        for (int n=0;n<=this.outputNeurons.length-1;n++){
            OutputNeuron thisOutputNeuron = this.outputNeurons[n];
            thisOutputNeuron.lossFunctionsDerivatives.put("mse",thisOutputNeuron.derivativeMSEWRTActivation());
            thisOutputNeuron.lossFunctionsDerivatives.put("bce",thisOutputNeuron.derivativeBCEWRTActivation());
            thisOutputNeuron.lossFunctionsDerivatives.put("bceWithLogits",thisOutputNeuron.derivativeBCELogitsWRTActivation());
        }
    }

    // getters and setters for the loss function name that is used
    // as a key in the loss derivatives hashmap
    public String getLossFunction(){
        return this.lossFunction;
    }

    public void setLossFunction(String lossFunctionName){
        this.lossFunction = lossFunctionName;
    }

    // updating all the weights of every neuron in the output layer
    public void updateOutputNeuronsWeights(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateWeights(i);
        }
    }

    // updating all the weighted input for all the output neurons
    public void updateOutputNeuronsWeightedInputs(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateWeightedInputValue();
        }
    }

    // updating the error signal for each output neuron
    // error signal is , for example:
    // (dz2/da1)*(da2/dz2)*(dL/da2)
    // w2 * [a2*(1-a2)] * [2(a2-y)]
    public void updateOutputNeuronsErrorSignals(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateErrorSignals();
        }
    }

    // updating the biases
    public void updateOutputNeuronsBiases(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateBias(i);
        }
    }

    // setting the true values
    public void updateOutputLayerTrueValues(double[] trueValues){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].trueValue = trueValues[i];
        }
    }

    // incrementing the time step
    public void incrementTimeStep(){
        this.timeStep++;
    }

    // reseting the gradients to zero
    public void zeroGrad(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].weightsAccumulatedGradients.clear();
            this.outputNeurons[i].biasAccumulatedGradients = 0;
        }
    }

    public void resetMomentumes(){
        this.firstMomentumW = new HashMap<>();
        this.secondMomentumW = new HashMap<>();
        this.firstMomentumB = new HashMap<>();
        this.secondMomentumB = new HashMap<>();
    }

    // updating the gradients
    public void updateGradients(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateErrorSignals();
            this.outputNeurons[i].updateAllWeightsGradients();
            this.outputNeurons[i].updateBiasGradient();
        }
    }

    // updating the params (i.e, weights , biases)
    public void updateParameters(){
        for (int i=0;i<=this.outputNeurons.length-1;i++){
            this.outputNeurons[i].updateWeights(i);
            this.outputNeurons[i].updateBias(i);
        }
    }
}