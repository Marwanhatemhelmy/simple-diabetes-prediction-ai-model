class InputNeuron{
    public double inputValue;
    public InputNeuron(){
        // defult input neuron value to zero
        // * IMPORTANT NOTE *
        // any weight that comes after a 0 value input neuron , so any weight in the first hidden layer
        // connected to an input neuron with 0 value , wont get updated unless this value is changed
        // in up coming rows with non-zero values
        this.inputValue = 0;
    }
}