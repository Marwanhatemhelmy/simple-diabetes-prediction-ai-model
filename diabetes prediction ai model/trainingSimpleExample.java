import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class trainingSimpleExample {
    public static void main(String[] args) {
        int inputLayerLength = 8;
        int hiddenLayersLength = 1;
        int hiddenLayerLength = 28;
        int outputLayerLength = 1;
        int batchSize = 192;
        double learningRate = 0.03;
        // setting the neural network
        NeuralNetwork nn = new NeuralNetwork(inputLayerLength, hiddenLayersLength, hiddenLayerLength, outputLayerLength, batchSize, learningRate);
        // setting the model
        NeuralNetworkModel nnm = new NeuralNetworkModel(nn);
        nnm.setUpTheModel();
        // setting the loss function
        new LossFunctions(nn.outputLayer).BCELoss();
        // the path for the csv dataset
        String filePath = "diabetes.csv";

        // using mini-batch approach
        // batch size = dataset size/4 = 768/4 = 192
        // training time
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            int epoch = 0;
            
            // setting a number epochs
            while (epoch < 500) {
                nn.zeroGradients();

                String line;
                int rowIndex = 0;

                BufferedReader currentReader = new BufferedReader(new FileReader(filePath));
                while ((line = currentReader.readLine()) != null) {
                    if (rowIndex == 0) {
                        rowIndex++;
                        continue;
                    }

                    String[] values = line.split(",");
                    double[] inputs = new double[8];
                    double[] trueValues = new double[1];

                    for (int v = 0; v <= inputs.length-1; v++) {
                        inputs[v] = Double.parseDouble(values[v]);
                        if (v==7){
                            trueValues[0] = (Double.parseDouble(values[8]));
                        }
                    }
                    
                    nn.inputLayer.updateInputs(inputs);
                    nn.outputLayer.updateOutputLayerTrueValues(trueValues);
                    nn.forward();
                    nn.outputLayer.updateOutputNeuronsLossDerivativesHashMap();
                    nn.updateGradients();
                    if (rowIndex%nn.batchSize==0&&rowIndex!=0){
                        nn.updateParameters();
                        nn.incrementTimeStep();
                        nn.zeroGradients();
                    }

                    rowIndex++;
                } 
                epoch++;
                currentReader.close();
                if (epoch%100==0){
                    System.out.println("Epoch completed: " + epoch);
                }
            }
            //##########################################################//#endregion
            // testing time
            try (BufferedReader testReader = new BufferedReader(new FileReader(filePath))) {
                int correctAnswers = 0;
                int wrongAnswers = 0;
                String line;
                int rowIndex = 0;
        
                while ((line = testReader.readLine()) != null) {
                    if (rowIndex >= 1) {
                        String[] values = line.split(",");
                        double[] inputs = new double[8];
                        double[] trueValues = new double[1];

                        for (int v = 0; v <= inputs.length-1; v++) {
                            inputs[v] = Double.parseDouble(values[v]);
                            if (v==7){
                                trueValues[0] = Double.parseDouble(values[8]);
                            }
                        }
                        int diabetesState=0;
                        nn.inputLayer.updateInputs(inputs);
                        nn.forward();
                        if (nn.outputLayer.outputNeurons[0].sigmoidActivationFunction()>0.5){
                            diabetesState = 1;
                        }
                        if (diabetesState == trueValues[0]){
                            correctAnswers++;
                        }else{
                            wrongAnswers++;
                        }
                    }
                    rowIndex++;
                }
                System.out.println(nn.outputLayer.outputNeurons[0].weights[0]);
                System.out.println("correct answers: "+Integer.toString(correctAnswers));
                System.out.println("wrong answers: "+Integer.toString(wrongAnswers));
                System.out.println("accuracy: "+Double.toString((double)((double)correctAnswers/((double)correctAnswers+(double)wrongAnswers))*100)+"%");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
