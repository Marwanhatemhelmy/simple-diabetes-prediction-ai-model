// this class is for manually setting the layers with there parameters
public class NeuralNetwork {
    public int inputLayerLegth;
    public int hiddenLayersLength;
    public int hiddenLayerLength;
    public int outputLayerLength;
    public double batchSize;
    public double learningRate;
    public InputLayer inputLayer = new InputLayer();
    public HiddenLayers hiddenLayers = new HiddenLayers();
    public OutputLayer outputLayer = new OutputLayer();
    public NeuralNetwork(int inputLayerLegth, int hiddenLayersLength, int hiddenLayerLength, int outputLayerLength,double batchSize, double learningRate){
        // this.inputLayer = new InputLayer();
        // this.hiddenLayers = new HiddenLayers();
        // this.outputLayer = new OutputLayer();
        this.inputLayerLegth = inputLayerLegth;
        this.hiddenLayersLength = hiddenLayersLength;
        this.hiddenLayerLength = hiddenLayerLength;
        this.outputLayerLength = outputLayerLength;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.hiddenLayers.hiddenLayers = new HiddenLayer[this.hiddenLayersLength];
        this.hiddenLayers.learningRate = learningRate;
        this.hiddenLayers.outputLayer = this.outputLayer;
    }
    // setting up the input layer with initial inputs
    public void setUpInputLayer(double[] inputs){
        this.inputLayer.inputNeurons = new InputNeuron[this.inputLayerLegth];
        for (int i=0;i<=this.inputLayerLegth-1;i++){
            InputNeuron thisInputNeuron = new InputNeuron();
            thisInputNeuron.inputValue = inputs[i];
            this.inputLayer.inputNeurons[i] = thisInputNeuron;
        }
    }

    // setting a specific hidden layer , with a specific index to it
    // weights and biases must be passed
    public void setUpHiddenLayer(int hiddenLayerIndex, double[][] weights, double[] bias){
        this.hiddenLayers.hiddenLayers[hiddenLayerIndex] = new HiddenLayer();
        this.hiddenLayers.hiddenLayers[hiddenLayerIndex].hiddenNeurons = new HiddenNeuron[this.hiddenLayerLength];
        for (int i=0;i<=this.hiddenLayerLength-1;i++){
            if (hiddenLayerIndex == 0){
                HiddenNeuron thisHiddenNeuron = new HiddenNeuron(this.inputLayer.inputNeurons,this.outputLayer,this.hiddenLayers);
                thisHiddenNeuron.weights = weights[i];
                thisHiddenNeuron.bias = bias[i];
                this.hiddenLayers.hiddenLayers[hiddenLayerIndex].hiddenNeurons[i] = thisHiddenNeuron;
            }else{
                HiddenNeuron thisHiddenNeuron = new HiddenNeuron(this.hiddenLayers.hiddenLayers[hiddenLayerIndex-1].hiddenNeurons,this.outputLayer,this.hiddenLayers);
                thisHiddenNeuron.weights = weights[i];
                thisHiddenNeuron.bias = bias[i];
                this.hiddenLayers.hiddenLayers[hiddenLayerIndex].hiddenNeurons[i] = thisHiddenNeuron;
            }
        }
    }

    // setting up the output layer
    public void setUpOutputLayer(double[][] weights, double[] bias){
        OutputNeuron[] thisOutPutLayerNeurons = new OutputNeuron[this.outputLayerLength];
        OutputLayer thisOutputLayer = new OutputLayer(thisOutPutLayerNeurons);
        this.outputLayer = thisOutputLayer;
        this.hiddenLayers.outputLayer = thisOutputLayer;
        for (int i=0;i<=this.outputLayerLength-1;i++){
            OutputNeuron thisOutputNeuron = new OutputNeuron(this.hiddenLayers.hiddenLayers[this.hiddenLayersLength-1].hiddenNeurons,this.outputLayer,this.batchSize,this.learningRate);
            thisOutputNeuron.weights = weights[i];
            thisOutputNeuron.bias = bias[i];
            thisOutPutLayerNeurons[i] = thisOutputNeuron;
        }
    }

    // updating the gradient for hidden layers and and output layers
    public void updateGradients(){
        this.outputLayer.updateGradients();
        this.hiddenLayers.updateGradients();
    }

    // updating the parameters (i.e, weights,biases) , for output neurons
    // and hidden neurons
    public void updateParameters(){
        this.outputLayer.updateParameters();
        this.hiddenLayers.updateParameters();
    }

    // forward pass of the inputs
    // inputs => hidden layers => output layer
    public void forward(){
        this.hiddenLayers.updateHiddenNeuronsWeightedInputs();
        this.outputLayer.updateOutputNeuronsWeightedInputs();
    }
    
    // reseting the gradients to zero
    public void zeroGradients(){
        this.outputLayer.zeroGrad();
        this.hiddenLayers.zeroGrad();
    }

    // incrementing the time step by 1
    // to keep track of where we are
    public void incrementTimeStep(){
        this.outputLayer.incrementTimeStep();
        this.hiddenLayers.incrementTimeStep();
    }

    public static void main(String[] args){
        
    }
}
