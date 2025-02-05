// this class is to let you change between loss function , and choose the most siutable loss function
// for your model
public class LossFunctions {
    OutputLayer outputLayer;
    // needs the output layer to be passed to it
    public LossFunctions(OutputLayer outputLayer){
        this.outputLayer = outputLayer;
    }
    public double MSELoss(){
        this.outputLayer.setLossFunction("mse");
        return this.outputLayer.meanSquaredError();
    }
    public double BCELoss(){
        this.outputLayer.setLossFunction("bce");
        return this.outputLayer.binaryCrossEntropy();
    }
    public void BCEWithLogitsLoss(){
        this.outputLayer.setLossFunction("bceWithLogits");
    }
}
