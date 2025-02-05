class InputLayer{
    InputNeuron[] inputNeurons;
    public InputLayer(){}
    public InputLayer(InputNeuron[] inputNeurons){
        this.inputNeurons = inputNeurons;
    }

    public void updateInputs(double[] newInputs){
        boolean leaveNewInputs = true;
        if (newInputs.length == 2){
            for (int i=0;i<=newInputs.length-1;i++){
                if (newInputs[i]>1||newInputs[i]<0){
                    leaveNewInputs = false;
                    break;
                }
            }
        }
        //########################################//#endregion
        double[] normalizedData = zScoreNormalize(newInputs);

        // extraNormalizedData is and array with the length of the normalizedData + 1
        // because if the length of the new inputs was 2 it would return -0 and 0 no matter what is the
        // original values
        double[] extraNormalizedData = new double[normalizedData.length+1];
        for (int n=0;n<=newInputs.length-1;n++){
            extraNormalizedData[n]=normalizedData[n];
        }
        if (leaveNewInputs){extraNormalizedData=normalizedData;}
        for (int i=0;i<=this.inputNeurons.length-1;i++){
            this.inputNeurons[i].inputValue = extraNormalizedData[i];
        }
    }

    // Function to normalize a dataset using Z-score normalization
    public static double[] zScoreNormalize(double[] data) {
        double mean = calculateMean(data);
        double stdDev = calculateStdDev(data, mean);

        double[] normalizedData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normalizedData[i] = (data[i] - mean) / stdDev;
        }
        return normalizedData;
    }

    // Function to calculate the mean of an array
    public static double calculateMean(double[] data) {
        double sum = 0.0;
        for (double num : data) {
            sum += num;
        }
        return sum / data.length;
    }

    // Function to calculate the standard deviation of an array
    public static double calculateStdDev(double[] data, double mean) {
        double sum = 0.0;
        for (double num : data) {
            sum += Math.pow(num - mean, 2);
        }
        return Math.sqrt(sum / data.length);
    }
}