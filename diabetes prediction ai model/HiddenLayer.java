import java.util.HashMap;

class HiddenLayer{
    public HiddenNeuron[] hiddenNeurons;
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> firstMomentumW = new HashMap<>();
    // {0:{0:{0:0.0}}}
    // t  ni  wi wv
    // t = time step
    // ni = neuron index
    // wi = weight index
    // wv = weight value
    // example:
    // {1:{0:{0:0.4,1:0.7}}}
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> secondMomentumW = new HashMap<>();
    public HashMap<Integer, HashMap<Integer, Double>> firstMomentumB = new HashMap<>();
    public HashMap<Integer, HashMap<Integer, Double>> secondMomentumB = new HashMap<>();
    public HiddenLayer(){}
    public HiddenLayer(HiddenNeuron[] hiddenNeurons){
        this.hiddenNeurons = hiddenNeurons;
    }
}