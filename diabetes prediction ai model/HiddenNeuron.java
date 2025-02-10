import java.util.HashMap;

class HiddenNeuron{
    public double bias;
    public double[] weights;
    public InputNeuron[] inputNeurons;
    public HiddenNeuron[] hiddenNeurons;
    public double weightedInputValue = 0;
    public OutputLayer outputLayer;
    public HiddenLayers hiddenLayers;
    public HashMap<Integer, Double> errorSignal;
    public HashMap<Integer,Double> weightsAccumulatedGradients = new HashMap<Integer,Double>();
    public double biasAccumulatedGradients = 0;
    public double beta1 = 0.9;
    public double beta2 = 0.999;
    public double epsilion = 1e-8;

    public HiddenNeuron(InputNeuron[] inputNeurons , OutputLayer outputLayer, HiddenLayers hiddenLayers){
        this.bias = 0;
        this.inputNeurons = inputNeurons;
        this.weights = new double[this.inputNeurons.length];
        this.outputLayer = outputLayer;
        this.errorSignal = new HashMap<Integer, Double>();
        this.hiddenLayers = hiddenLayers;
        this.updateWeightedInputValue();
    }

    public HiddenNeuron(HiddenNeuron[] hiddenNeurons , OutputLayer outputLayer, HiddenLayers hiddenLayers){
        this.bias = 0;
        this.hiddenNeurons = hiddenNeurons;
        this.weights = new double[this.hiddenNeurons.length];
        this.outputLayer = outputLayer;
        this.errorSignal = new HashMap<Integer, Double>();
        this.hiddenLayers = hiddenLayers;
        this.updateWeightedInputValue();
    }

    // z
    public void updateWeightedInputValue(){
        this.weightedInputValue = 0;
        for (int i = 0; i<=weights.length-1; i++){
            if (this.inputNeurons!=null){
                double thisWeightedSum = this.weights[i] * this.inputNeurons[i].inputValue;
                this.weightedInputValue += thisWeightedSum;
            }else if(this.hiddenNeurons!=null){
                double thisWeightedSum = this.weights[i] * this.hiddenNeurons[i].sigmoidActivationFunction();
                this.weightedInputValue += thisWeightedSum;
            }
        }
        this.weightedInputValue += this.bias;
    }

    // a
    public double sigmoidActivationFunction(){
        return 1.0/(1.0 + Math.exp(-this.weightedInputValue));
    }

    // (da[i]/dz[i])
    public double derivativeSigmoid(){
        double activation = this.sigmoidActivationFunction();
        return activation*(1-activation);
    }

    // (dz[i]/dw[i])
    public double derivativeWeightedInputWRTWeight(int weightIndex){
        if (this.hiddenNeurons != null){
            return this.hiddenNeurons[weightIndex].sigmoidActivationFunction();
        }
        return this.inputNeurons[weightIndex].inputValue;
    }

    // (dz[i-1]/da[i])
    public double derivativeWeightedInputWRTInput(int inputNeuronIndex){
        return this.weights[inputNeuronIndex];
    }

    // (dL/dw) , (i.e, (d,MSELoss/dw2))
    public double derivativeLossWRTWeight(int weightIndex, double previousGradients){
        return this.derivativeWeightedInputWRTWeight(weightIndex)*this.derivativeSigmoid()*previousGradients;
    }

    // (dL/db)
    public double derivativeLossWRTBias(double previousGradients){
        return this.derivativeSigmoid()*previousGradients;
    }

    //###############################################//#endregion

    // updating the error signal for this neuron so that it's easily attained for the next layer neurons
    // to facilitate the calculation of the next layer neurons gradients
    public void updateErrorSignal(int weightIndex, double errorSignal){
        this.errorSignal.put(weightIndex, errorSignal);
    }

    // update the gradient for a given weight
    public void updateWeightsGradients(int neuronIndex, int weightIndex, HashMap<Integer,Double> previousGradients){
        if (this.weightsAccumulatedGradients.get(weightIndex)!=null){
            this.weightsAccumulatedGradients.put(weightIndex,
             this.weightsAccumulatedGradients.get(weightIndex)+this.derivativeLossWRTWeight(weightIndex,previousGradients.get(neuronIndex).doubleValue()));
        }else{
            this.weightsAccumulatedGradients.put(weightIndex,this.derivativeLossWRTWeight(weightIndex,previousGradients.get(neuronIndex).doubleValue()));
        }
    }

    // update the gradient for the bias
    public void updateBiasGradient(double previousGradients){
        this.biasAccumulatedGradients += (this.derivativeSigmoid()*this.derivativeLossWRTBias(previousGradients));
    }

    // update the gradients for all the weights
    public void updateAllWeightsGradients(int neuronIndex, HashMap<Integer,Double> previousGradients){
        for (int w=0;w<=this.weights.length-1;w++){
            // calling the function for every weight
            this.updateWeightsGradients(neuronIndex, w, previousGradients);
        }
    }

    // update all the weights
    // this function has been simplified using chat gpt, but the original idea behide the algorithm used
    // in it has been esstablished by me , bases on the reseaches i made about adam optimizer
    // this function can be modified in the future to fit with future optimizers,
    // currently it only supports adam optimizer
    // * the same goes with the updateBias function *
    public void updateWeights(HiddenLayer thisHiddenLayer, int neuronIndex, HashMap<Integer, Double> previousLayerNeuronsErrorSignals, int timeStep){
        for (int w = 0; w < this.weights.length; w++) {

            double derivativeLossWRTWeight = (1.0/this.outputLayer.batchSize)*this.weightsAccumulatedGradients.get(w).doubleValue();

            // Retrieve previous momentums (default to 0 if first time step)
            double prevFirstMomentumW = timeStep > 1 
                ? thisHiddenLayer.firstMomentumW.get(timeStep - 1).getOrDefault(neuronIndex, new HashMap<>()).getOrDefault(w, 0.0) 
                : 0.0;
        
            double prevSecondMomentumW = timeStep > 1 
                ? thisHiddenLayer.secondMomentumW.get(timeStep - 1).getOrDefault(neuronIndex, new HashMap<>()).getOrDefault(w, 0.0) 
                : 0.0;
        
            // Calculate current momentums
            double thisFirstMomentum = this.beta1 * prevFirstMomentumW + 0.1 * derivativeLossWRTWeight;
            double thisSecondMomentum = this.beta2 * prevSecondMomentumW + 0.001 * Math.pow(derivativeLossWRTWeight, 2);
        
            // Bias-corrected momentums
            double mHat = thisFirstMomentum / (1.0 - Math.pow(this.beta1, timeStep));
            double vHat = thisSecondMomentum / (1.0 - Math.pow(this.beta2, timeStep));
        
            // Update momentums in the hash maps
            thisHiddenLayer.firstMomentumW
                .computeIfAbsent(timeStep, k -> new HashMap<>())
                .computeIfAbsent(neuronIndex, k -> new HashMap<>())
                .put(w, thisFirstMomentum);
        
            thisHiddenLayer.secondMomentumW
                .computeIfAbsent(timeStep, k -> new HashMap<>())
                .computeIfAbsent(neuronIndex, k -> new HashMap<>())
                .put(w, thisSecondMomentum);
        
            // Update weight
            this.weights[w] -= this.hiddenLayers.learningRate * (mHat / (Math.sqrt(vHat) + this.epsilion));
        }
    }

    // updating the bias
    public void updateNeuronsBiases(HiddenLayer thisHiddenLayer, int neuronIndex, HashMap<Integer,Double> previousLayerNeuronsErrorSignals, int timeStep){
        
        double derivativeLossWRTBias = (1.0/this.outputLayer.batchSize)*this.biasAccumulatedGradients;

        double previousFirstMomentumB = 0.0;
        double previousSecondMomentumB = 0.0;
        if (timeStep!=1){ 
            previousFirstMomentumB = thisHiddenLayer.firstMomentumB.get(timeStep-1).get(neuronIndex);
            previousSecondMomentumB = thisHiddenLayer.secondMomentumB.get(timeStep-1).get(neuronIndex);
        }
        // calculate current momentums for bias
        double thisFirstMomentumB = this.beta1*previousFirstMomentumB+(1-this.beta1)*derivativeLossWRTBias;
        double thisSecondMomentumB = this.beta2*previousSecondMomentumB+(1-this.beta2)*Math.pow(derivativeLossWRTBias,2);
        // calculated bias corrections
        double mHat = thisFirstMomentumB/(1.0-Math.pow(this.beta1,timeStep));
        double vHat = thisSecondMomentumB/(1.0-Math.pow(this.beta2,timeStep));

        HashMap<Integer, Double> thisFirstMomentumBHashMap = new HashMap<Integer, Double>();
        HashMap<Integer, Double> thisSecondMomentumBHashMap = new HashMap<Integer, Double>();

        // adding this momentums to the hash map to be feeded for the next layer's neurons to calculate
        // their momentums
        thisFirstMomentumBHashMap.put(neuronIndex,thisFirstMomentumB);
        thisSecondMomentumBHashMap.put(neuronIndex,thisSecondMomentumB);
        if (thisHiddenLayer.firstMomentumB.get(timeStep)!=null){
            thisHiddenLayer.firstMomentumB.get(timeStep).put(neuronIndex,thisFirstMomentumB);
            thisHiddenLayer.secondMomentumB.get(timeStep).put(neuronIndex,thisSecondMomentumB);
        }else{
            thisHiddenLayer.firstMomentumB.put(timeStep, thisFirstMomentumBHashMap);
            thisHiddenLayer.secondMomentumB.put(timeStep, thisSecondMomentumBHashMap);
        }
        // updating the bias
        this.bias = this.bias - this.hiddenLayers.learningRate * (mHat/(Math.sqrt(vHat)+this.epsilion));
    }
}