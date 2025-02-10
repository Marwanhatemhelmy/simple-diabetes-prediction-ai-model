class InputLayer{
    InputNeuron[] inputNeurons;
    public InputLayer(){}
    public InputLayer(InputNeuron[] inputNeurons){
        this.inputNeurons = inputNeurons;
    }

    // this function updates every input neuron value
    public void updateInputs(double[] newInputs){
        for (int i=0;i<=this.inputNeurons.length-1;i++){
            // i is the index of the input neuron
            this.inputNeurons[i].inputValue = newInputs[i];
        }
    }
}