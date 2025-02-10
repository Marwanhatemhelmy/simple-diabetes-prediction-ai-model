import java.util.Random;
// this class automatically sets up the whole model with it's layers ,input , hidden and output layers
public class NeuralNetworkModel {
    NeuralNetwork nn;
    Random random = new Random();

    // need to pass it the neural network you initialized earlier
    public NeuralNetworkModel(NeuralNetwork nn){
        this.nn = nn;
    }

    // the function that is resposible for seting up the model layers
    public void setUpTheModel(){
        // intializing the input layer
        double[] inputNeuronsValues = new double[this.nn.inputLayerLegth];
        for (int w=0;w<=inputNeuronsValues.length-1;w++){
            inputNeuronsValues[w] = 1;
        }
        this.nn.setUpInputLayer(inputNeuronsValues);

        // intializing the hidden layers
        for (int h=0;h<=this.nn.hiddenLayersLength-1;h++){
            double[][] thisHiddenLayerWeights;
            if (h==0){
                thisHiddenLayerWeights = new double[this.nn.hiddenLayerLength][this.nn.inputLayerLegth];
                for (int n=0;n<=this.nn.hiddenLayerLength-1;n++){
                    for (int i=0;i<=this.nn.inputLayerLegth-1;i++){
                        thisHiddenLayerWeights[n][i]= this.random.nextDouble() * 2 - 1; 
                    }
                }
            }else{
                thisHiddenLayerWeights = new double[this.nn.hiddenLayerLength][this.nn.hiddenLayerLength];
                for (int n=0;n<=this.nn.hiddenLayerLength-1;n++){
                    for (int i=0;i<=this.nn.hiddenLayerLength-1;i++){
                        thisHiddenLayerWeights[n][i]= this.random.nextDouble() * 2 - 1; 
                    }
                }
            }
            this.nn.setUpHiddenLayer(h, thisHiddenLayerWeights, new double[this.nn.hiddenLayerLength]);
        }
        // intializing the output layer
        double[][] outputLayerWeights = new double[this.nn.outputLayerLength][this.nn.hiddenLayerLength];
        for (int on=0;on<=this.nn.outputLayerLength-1;on++){
            // on = output neuron index
            for (int ow=0;ow<=this.nn.hiddenLayerLength-1;ow++){
                outputLayerWeights[on][ow] = this.random.nextDouble() * 2 - 1; 
            }
        }
        this.nn.setUpOutputLayer(outputLayerWeights, new double[this.nn.outputLayerLength]);
    }
}
