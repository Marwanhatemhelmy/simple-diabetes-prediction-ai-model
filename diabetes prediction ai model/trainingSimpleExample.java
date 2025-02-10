import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class trainingSimpleExample {
    public static void main(String[] args) {
        int inputLayerLength = 8;
        int hiddenLayersLength = 1;
        int hiddenLayerLength = 128;
        int outputLayerLength = 1;
        int batchSize = 700;
        double learningRate = 0.003;
        // setting the neural network
        NeuralNetwork nn = new NeuralNetwork(inputLayerLength, hiddenLayersLength, hiddenLayerLength, outputLayerLength, batchSize, learningRate);
        // setting the model
        NeuralNetworkModel nnm = new NeuralNetworkModel(nn);
        nnm.setUpTheModel();
        // setting the loss function
        new LossFunctions(nn.outputLayer).BCELoss();
        // the path for the csv dataset
        String filePath = "diabetes.csv";

        double[][] data = new double[700][8];
        double[] corrections = new double[700];

        double[][] validationData = new double[68][8];
        double[] validationCorrections = new double[68];

        // reading the csv dataset file
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int rowIndex = 0;

            // true for test / false for validation
            boolean testOrValidation = true;

            // reading each line (row)
            while ((line = br.readLine()) != null) {
                // canceling first raw as it's lables row
                if (rowIndex == 0) {
                    // but also increamenting as it's still considered a row in the dataset
                    rowIndex++;
                    continue;
                }

                if (rowIndex==701){
                    testOrValidation = false;
                }

                // spliting the row
                String[] values = line.split(",");
                // setting the new inputs to train the model on
                double[] inputs = new double[8];
                // setting the true values to calculate the loss function on
                double[] trueValues = new double[]{Double.parseDouble(values[8])};

                for (int v = 0; v <= inputs.length-1; v++) {
                    inputs[v] = (Double.parseDouble(values[v]));
                }

                // classifying the data to training and validation data
                if (testOrValidation){
                    data[rowIndex-1] = inputs;
                    corrections[rowIndex-1] = trueValues[0];
                }else{
                    validationData[rowIndex-701] = inputs;
                    validationCorrections[rowIndex-701] = trueValues[0];
                }
                rowIndex++;
            }

            // normalize the data using z-score normalizing technice
            ZScoreNormalization zsn = new ZScoreNormalization();
            zsn.normalize(data);
            zsn.normalize(validationData);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // training segment
        int epoch = 0;
        while (epoch<150){
            nn.zeroGradients();
            for (int i=0;i<=data.length-1;i++) {
                double[] inputs = data[i];
                double[] trueValues = new double[1];
                trueValues[0] = corrections[i];

                nn.inputLayer.updateInputs(inputs);
                nn.outputLayer.updateOutputLayerTrueValues(trueValues);
                nn.forward();
                nn.updateGradients();
                if (i%nn.batchSize == 0 && i!=0 || i==data.length-1){
                    nn.updateParameters();
                    nn.incrementTimeStep();
                    nn.zeroGradients();
                }
            }
            epoch++;
            if (epoch%100==0){
                System.out.println("Epoch completed: " + epoch);
            }
        }

        // validation segment
        int correctAnswers = 0;
        int wrongAnswers = 0;
        for (int t=0;t<=validationData.length-1;t++){
            double[] inputs = validationData[t];
            double[] trueValues = new double[1];
            trueValues[0] = validationCorrections[t];

            nn.inputLayer.updateInputs(inputs);
            nn.outputLayer.updateOutputLayerTrueValues(trueValues);
            nn.forward();

            int diabetesState=0;
            if (nn.outputLayer.outputNeurons[0].sigmoidActivationFunction()>0.5){
                diabetesState = 1;
            }
            if (diabetesState == trueValues[0]){
                correctAnswers++;
            }else{
                wrongAnswers++;
            }
        }

        // outputing the number of correct answers and wrong answers and the accuracy in percentage
        System.out.println(nn.outputLayer.outputNeurons[0].weights[0]);
        System.out.println("correct answers: "+Integer.toString(correctAnswers));
        System.out.println("wrong answers: "+Integer.toString(wrongAnswers));
        System.out.println("accuracy: "+Double.toString((double)((double)correctAnswers/((double)correctAnswers+(double)wrongAnswers))*100)+"%");
    }
}
