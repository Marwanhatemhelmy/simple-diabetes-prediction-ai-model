import java.util.HashMap;

class OutputNeuron {
    public double bias;
    public double[] weights;
    public HiddenNeuron[] hiddenNeurons;
    public double weightedInputValue;
    public double trueValue;
    public OutputLayer outputLayer;
    public double batchSize;
    public double learningRate;
    public HashMap<Integer, Double> errorSignals;
    public HashMap<Integer,Double> weightsAccumulatedGradients = new HashMap<Integer,Double>();
    public double biasAccumulatedGradients = 0;
    public HashMap<String,Double> lossFunctionsDerivatives = new HashMap<String,Double>();

    public OutputNeuron(HiddenNeuron[] hiddenNeurons , OutputLayer outputLayer , double batchSize, double learningRate){
        this.bias = 0;
        this.hiddenNeurons = hiddenNeurons;
        this.weights = new double[this.hiddenNeurons.length];
        this.outputLayer = outputLayer;
        this.trueValue = 0;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.errorSignals = new HashMap<Integer, Double>();
        this.updateWeightedInputValue();
    }

    public void updateWeightedInputValue(){
        this.weightedInputValue = 0;
        for (int i = 0; i<=weights.length-1; i++){
            double thisWeightedSum = this.weights[i] * this.hiddenNeurons[i].sigmoidActivationFunction();
            this.weightedInputValue += thisWeightedSum;
        }
        this.weightedInputValue += this.bias;
    }

    public double sigmoidActivationFunction(){
        return 1.0/(1.0 + Math.exp(-this.weightedInputValue));
    }
    
    public double derivativeSigmoid(){
        double activation = this.sigmoidActivationFunction();
        return activation*(1-activation);
    }

    public double derivativeMSEWRTActivation(){
        return 2*(this.sigmoidActivationFunction()-this.trueValue);
    }

    
    public double derivativeBCELogitsWRTActivation(){
        return (1/(1+Math.exp(-this.sigmoidActivationFunction())))-this.trueValue;
    }

    public double derivativeBCEWRTActivation(){
        return -(this.trueValue/this.sigmoidActivationFunction())+((1-this.trueValue)/(1-this.sigmoidActivationFunction()));
    }

    public double derivativeWeightedInputWRTWeight(int weightIndex){
        return this.hiddenNeurons[weightIndex].sigmoidActivationFunction();
    }

    public double derivativeLossWRTWeight(int weightIndex){
        double lossWRTActivation = this.lossFunctionsDerivatives.get(this.outputLayer.getLossFunction());
        return this.derivativeWeightedInputWRTWeight(weightIndex)*this.derivativeSigmoid()*lossWRTActivation;
    }

    public double derivativeLossWRTBias(){
        double lossWRTActivation = this.lossFunctionsDerivatives.get(this.outputLayer.getLossFunction());
        return this.derivativeSigmoid()*lossWRTActivation;
    }

    public void updateWeightsGradients(int weightIndex){
        if (this.weightsAccumulatedGradients.get(weightIndex)!=null){
            this.weightsAccumulatedGradients.put(weightIndex,this.weightsAccumulatedGradients.get(weightIndex).doubleValue()+this.derivativeLossWRTWeight(weightIndex));
        }else{
            this.weightsAccumulatedGradients.put(weightIndex,this.derivativeLossWRTWeight(weightIndex));
        }
    }

    public void updateBiasGradient(){
        this.biasAccumulatedGradients += this.derivativeLossWRTBias();
    }

    public void updateAllWeightsGradients(){
        for(int w=0;w<=this.weights.length-1;w++){
            this.updateWeightsGradients(w);
        }
    }

    public void updateWeights(int neuronIndex){
        for (int i = 0; i < this.weights.length; i++) {
            double derivativeLossWRTWeight = (1.0/this.batchSize)*this.weightsAccumulatedGradients.get(i);
            
            // Retrieve previous momentums (default to 0 if first time step)
            double prevFirstMomentum = this.outputLayer.timeStep > 1 
                ? this.outputLayer.firstMomentumW.get(this.outputLayer.timeStep - 1).getOrDefault(neuronIndex, new HashMap<>()).getOrDefault(i, 0.0) 
                : 0.0;
        
            double prevSecondMomentum = this.outputLayer.timeStep > 1 
                ? this.outputLayer.secondMomentumW.get(this.outputLayer.timeStep - 1).getOrDefault(neuronIndex, new HashMap<>()).getOrDefault(i, 0.0) 
                : 0.0;
        
            // Calculate current momentums
            double currFirstMomentum = 0.9 * prevFirstMomentum + 0.1 * derivativeLossWRTWeight;
            double currSecondMomentum = 0.999 * prevSecondMomentum + 0.001 * Math.pow(derivativeLossWRTWeight, 2);
        
            // Bias-corrected momentums
            double mHat = currFirstMomentum / (1.0 - Math.pow(0.9, this.outputLayer.timeStep));
            double vHat = currSecondMomentum / (1.0 - Math.pow(0.999, this.outputLayer.timeStep));
        
            // Update momentums in the hash maps
            this.outputLayer.firstMomentumW
                .computeIfAbsent(this.outputLayer.timeStep, k -> new HashMap<>())
                .computeIfAbsent(neuronIndex, k -> new HashMap<>())
                .put(i, currFirstMomentum);
        
            this.outputLayer.secondMomentumW
                .computeIfAbsent(this.outputLayer.timeStep, k -> new HashMap<>())
                .computeIfAbsent(neuronIndex, k -> new HashMap<>())
                .put(i, currSecondMomentum);
        
            // Update weight
            this.weights[i] -= this.learningRate * (mHat / (Math.sqrt(vHat) + 1e-8));
        }
    }

    public void updateErrorSignals(){
        for (int i=0;i<=this.weights.length-1;i++){
            double lossWRTActivation = this.lossFunctionsDerivatives.get(this.outputLayer.getLossFunction());
            this.errorSignals.put(i, (lossWRTActivation*this.derivativeSigmoid()*this.weights[i]));
        }
    }

    public void updateBias(int i){
        double derivativeLossWRTBias = (1.0/this.batchSize)*(this.biasAccumulatedGradients);

        double previousFirstMomentumB = 0.0;
        double previousSecondMomentumB = 0.0;
        if (this.outputLayer.timeStep!=1){ 
            previousFirstMomentumB = this.outputLayer.firstMomentumB.get(this.outputLayer.timeStep-1).get(i);
            previousSecondMomentumB = this.outputLayer.secondMomentumB.get(this.outputLayer.timeStep-1).get(i);
        }
        double thisFirstMomentumB = 0.9*previousFirstMomentumB+(1-0.9)*derivativeLossWRTBias;
        double thisSecondMomentumB = 0.999*previousSecondMomentumB+(1-0.999)*Math.pow(derivativeLossWRTBias,2);
        double mHat = thisFirstMomentumB/(1.0-Math.pow(0.9,this.outputLayer.timeStep));
        double vHat = thisSecondMomentumB/(1.0-Math.pow(0.999,this.outputLayer.timeStep));
        HashMap<Integer, Double> thisFirstMomentumBHashMap = new HashMap<Integer, Double>();
        HashMap<Integer, Double> thisSecondMomentumBHashMap = new HashMap<Integer, Double>();
        thisFirstMomentumBHashMap.put(i,thisFirstMomentumB);
        thisSecondMomentumBHashMap.put(i,thisSecondMomentumB);
        if (this.outputLayer.firstMomentumB.get(this.outputLayer.timeStep)!=null){
            this.outputLayer.firstMomentumB.get(this.outputLayer.timeStep).put(i,thisFirstMomentumB);
            this.outputLayer.secondMomentumB.get(this.outputLayer.timeStep).put(i,thisSecondMomentumB);
        }else{
            this.outputLayer.firstMomentumB.put(this.outputLayer.timeStep, thisFirstMomentumBHashMap);
            this.outputLayer.secondMomentumB.put(this.outputLayer.timeStep, thisSecondMomentumBHashMap);
        }
        this.bias = this.bias - this.learningRate * (mHat/(Math.sqrt(vHat)+Math.pow(10,-8)));
    }
}