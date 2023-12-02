package Model;

import java.text.DecimalFormat;
import java.util.LinkedList;

public class Neuron {
    private LinkedList<Connection> frontConnections, backConnections;
    private double value, bias, sumTotal, output, imputedError;
    private int layer;
    private final DecimalFormat formatDouble = new DecimalFormat("#.#####");

    //Constructor para neuronas de entrada o salida
    public Neuron(int layer, double value, double bias){
        this.layer = layer;
        this.value = value;
        this.bias = bias;
        this.sumTotal = 0;
        this.output = 0;
        this.imputedError = 0;
        this.frontConnections = new LinkedList<>();
        this.backConnections = new LinkedList<>();
    }

    //Constructor para las capas ocultas
    public Neuron(int layer, double bias){
        this.layer = layer;
        this.value = 0;
        this.bias = bias;
        this.sumTotal = 0;
        this.output = 0;
        this.value = 0;
        this.imputedError = 0;
        this.frontConnections = new LinkedList<>();
        this.backConnections = new LinkedList<>();
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getSumTotal() {
        return sumTotal;
    }

    public void setSumTotal(double sumTotal) {
        this.sumTotal = sumTotal;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getImputedError() {
        return imputedError;
    }

    public void setImputedError(double imputedError) {
        this.imputedError = imputedError;
    }

    public int getLayer() {
        return layer;
    }

    public void setLayer(int layer) {
        this.layer = layer;
    }
}
