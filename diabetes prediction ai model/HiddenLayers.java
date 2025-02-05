import java.util.HashMap;

class HiddenLayers {
    public HiddenLayer[] hiddenLayers;
    public OutputLayer outputLayer;
    public double learningRate;
    public int timeStep = 1;
    public HiddenLayers(){}
    public HiddenLayers(HiddenLayer[] hiddenLayers, OutputLayer outputLayer, double learningRate){
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.learningRate = learningRate;
        this.timeStep = 1;
    }

    // function that updates the accumulated gradients for every neuron in every hidden layer
    // and updates it's error signal for the up-coming layer's neurons
    public void updateGradients(){
        for (int i=this.hiddenLayers.length-1;i>=0;i--){
            HiddenLayer thisHiddenLayer = this.hiddenLayers[i];
            // checking if the previous layer is the output layer or a hidden layer
            if (i == this.hiddenLayers.length-1){
                
                OutputLayer previousLayer = this.outputLayer;
                // the error signal is the (wi)*(dai/dzi)*(wi+1)*(dai+1/dzi+1)*(dL/dai+1) , for example
                HashMap<Integer, Double> previousLayerNeuronsErrorSignals = new HashMap<>();
                for (int p=0;p<=previousLayer.outputNeurons.length-1;p++){
                    // p is the index of the previous neuron in the previous layer
                    OutputNeuron thisPreviousOutputNeuron = previousLayer.outputNeurons[p];
                    for (int wp=0;wp<=thisPreviousOutputNeuron.weights.length-1;wp++){
                        // wp is this weight in this previous layer neuron
                        if (p==0){
                            previousLayerNeuronsErrorSignals.put(wp,thisPreviousOutputNeuron.errorSignals.get(wp));
                        }else{
                            previousLayerNeuronsErrorSignals.put(wp, previousLayerNeuronsErrorSignals.get(wp)+thisPreviousOutputNeuron.errorSignals.get(wp));
                        }
                    }
                }
                for (int n=0;n<=thisHiddenLayer.hiddenNeurons.length-1;n++){
                    // n is the index of the neuron in it's layer
                    HiddenNeuron thisHiddenNeuron = thisHiddenLayer.hiddenNeurons[n];

                    // updating each weight's error signal
                    for (int w=0;w<=thisHiddenNeuron.weights.length-1;w++){
                        thisHiddenNeuron.updateErrorSignal(w, (thisHiddenNeuron.weights[w]*thisHiddenNeuron.derivativeSigmoid()*previousLayerNeuronsErrorSignals.get(n).doubleValue()));
                    }
                    // updating this hidden neuoron weights and biases gradients
                    thisHiddenNeuron.updateAllWeightsGradients(n,previousLayerNeuronsErrorSignals);
                    thisHiddenNeuron.updateBiasGradient(previousLayerNeuronsErrorSignals.get(n).doubleValue());
                }
            }else{
                HiddenLayer previousLayer = this.hiddenLayers[i+1];
                HashMap<Integer, Double> previousLayerNeuronsErrorSignals = new HashMap<>();
                for (int p=0;p<=previousLayer.hiddenNeurons.length-1;p++){
                    // p is the index of the previous neuron in the previous layer
                    HiddenNeuron thisPreviousHiddenNeuron = previousLayer.hiddenNeurons[p];
                    for (int wp=0;wp<=thisPreviousHiddenNeuron.weights.length-1;wp++){
                        // wp is this weight in this previous layer neuron
                        if (p==0){
                            previousLayerNeuronsErrorSignals.put(wp,thisPreviousHiddenNeuron.errorSignal.get(wp));
                        }else{
                            previousLayerNeuronsErrorSignals.put(wp, previousLayerNeuronsErrorSignals.get(wp)+thisPreviousHiddenNeuron.errorSignal.get(wp));
                        }
                    }  
                }
                
                for (int n=0;n<=thisHiddenLayer.hiddenNeurons.length-1;n++){
                    // n is the index of the neuron in it's layer
                    HiddenNeuron thisHiddenNeuron = thisHiddenLayer.hiddenNeurons[n];
                    
                    // updating each weight's error signal
                    for (int w=0;w<=thisHiddenNeuron.weights.length-1;w++){
                        thisHiddenNeuron.updateErrorSignal(w, (thisHiddenNeuron.weights[w]*thisHiddenNeuron.derivativeSigmoid()*previousLayerNeuronsErrorSignals.get(n).doubleValue()));
                    }
                    // updating this hidden neuoron weights and biases gradients
                    thisHiddenNeuron.updateAllWeightsGradients(n,previousLayerNeuronsErrorSignals);
                    thisHiddenNeuron.updateBiasGradient(previousLayerNeuronsErrorSignals.get(n).doubleValue());
                }
            }
        }
    }

    // this function update all the parameters (i.e, biase, weights) , for all the neurons in all
    // the hidden layers
    public void updateParameters(){
        for (int i=this.hiddenLayers.length-1;i>=0;i--){
            // i is this hidden layer index
            HiddenLayer thisHiddenLayer = this.hiddenLayers[i];
            if (i==this.hiddenLayers.length-1){
                OutputLayer previousLayer = this.outputLayer;
                // the error signal is the (wi)*(dai/dzi)*(wi+1)*(dai+1/dzi+1)*(dL/dai+1) , for example
                HashMap<Integer, Double> previousLayerNeuronsErrorSignals = new HashMap<>();
                for (int p=0;p<=previousLayer.outputNeurons.length-1;p++){
                    // p is the index of the previous neuron in the previous layer
                    OutputNeuron thisPreviousOutputNeuron = previousLayer.outputNeurons[p];
                    for (int wp=0;wp<=thisPreviousOutputNeuron.weights.length-1;wp++){
                        if (p==0){
                            previousLayerNeuronsErrorSignals.put(wp,thisPreviousOutputNeuron.errorSignals.get(wp));
                        }else{
                            previousLayerNeuronsErrorSignals.put(wp, previousLayerNeuronsErrorSignals.get(wp)+thisPreviousOutputNeuron.errorSignals.get(wp));
                        }
                    }
                }
                for (int n=0;n<=thisHiddenLayer.hiddenNeurons.length-1;n++){
                    // n is this hidden neuron
                    HiddenNeuron thisHiddenNeuron = thisHiddenLayer.hiddenNeurons[n];
                    // updating the weights and biases
                    thisHiddenNeuron.updateWeights(thisHiddenLayer, n, previousLayerNeuronsErrorSignals, this.timeStep);
                    thisHiddenNeuron.updateNeuronsBiases(thisHiddenLayer, n, previousLayerNeuronsErrorSignals, this.timeStep);
                }
            }else{
                HiddenLayer previousLayer = this.hiddenLayers[i+1];
                HashMap<Integer, Double> previousLayerNeuronsErrorSignals = new HashMap<>();
                for (int p=0;p<=previousLayer.hiddenNeurons.length-1;p++){
                    // p is the index of the previous neuron in the previous layer
                    HiddenNeuron thisPreviousHiddenNeuron = previousLayer.hiddenNeurons[p];
                    for (int wp=0;wp<=thisPreviousHiddenNeuron.weights.length-1;wp++){
                        if (p==0){
                            previousLayerNeuronsErrorSignals.put(wp,thisPreviousHiddenNeuron.errorSignal.get(wp));
                        }else{
                            previousLayerNeuronsErrorSignals.put(wp, previousLayerNeuronsErrorSignals.get(wp)+thisPreviousHiddenNeuron.errorSignal.get(wp));
                        }
                    }  
                }
                for (int n=0;n<=thisHiddenLayer.hiddenNeurons.length-1;n++){
                    // n is this hidden neuron
                    HiddenNeuron thisHiddenNeuron = thisHiddenLayer.hiddenNeurons[n];
                    // updating the weights and biases
                    thisHiddenNeuron.updateWeights(thisHiddenLayer, n, previousLayerNeuronsErrorSignals, this.timeStep);
                    thisHiddenNeuron.updateNeuronsBiases(thisHiddenLayer, n, previousLayerNeuronsErrorSignals, this.timeStep);
                }
            }
        }
    }

    // incrementing the time step to keep track of which time step the model is currently at
    // as we are using adam optimizer
    // if this code is modified in the future to be able to use multiple optimizers
    // this function would only be used in adam optimizer
    public void incrementTimeStep(){
        this.timeStep++;
    }

    // updating the weighted input values for every hidden neuron is essential for forward passing
    // after updating the input values in the input layer
    public void updateHiddenNeuronsWeightedInputs(){
        for (int i=0;i<=this.hiddenLayers.length-1;i++){
            for (int j=0;j<=this.hiddenLayers[i].hiddenNeurons.length-1;j++){
                // calling the function inside the neuron
                this.hiddenLayers[i].hiddenNeurons[j].updateWeightedInputValue();
            }
        }
    }

    // reseting the gradients to zero for biases and clearing the hashmap for weights to start a new cycle
    // and clearing accumulated gradients
    public void zeroGrad(){
        for (int i=0;i<=this.hiddenLayers.length-1;i++){
            HiddenLayer thisHiddenLayer = this.hiddenLayers[i];
            for (int n=0;n<=thisHiddenLayer.hiddenNeurons.length-1;n++){
                thisHiddenLayer.hiddenNeurons[n].weightsAccumulatedGradients.clear();
                thisHiddenLayer.hiddenNeurons[n].biasAccumulatedGradients = 0;
            }
        }
    }
}