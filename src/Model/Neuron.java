package Model;

import java.text.DecimalFormat;
import java.util.LinkedList;

public class Neuron {
    private LinkedList<Connection> frontConnections, backConnections;
    private double value, bias, sumTotal, output, imputedError;
    private int layer;
    private String name;
    private final DecimalFormat formatDouble = new DecimalFormat("#.#######");

    //Constructor para neuronas de entrada o salida
    public Neuron(String name, int layer, double value, double bias) {
        this.name = name;
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
    public Neuron(String name, int layer, double bias) {
        this.name = name;
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

    public LinkedList<Connection> getFrontConnections() {
        return frontConnections;
    }

    public void setFrontConnections(LinkedList<Connection> frontConnections) {
        this.frontConnections = frontConnections;
    }

    public LinkedList<Connection> getBackConnections() {
        return backConnections;
    }

    public void setBackConnections(LinkedList<Connection> backConnections) {
        this.backConnections = backConnections;
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

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String toString() {
        String limite = "|";
        return ("%s Neurona: %-5s Capa: %-3s Valor: %-10s    bias : %-10s      Suma Total : %-10s    Output : %-10s ErrorImputado:%-10s %s"
                .formatted(limite, this.name, this.layer, formatDouble.format(value), formatDouble.format(bias),
                        formatDouble.format(sumTotal), formatDouble.format(output), formatDouble.format(imputedError), limite));
    }

    public void printConnections() {
        System.out.printf("| Neurona : %-5s Conexiones -->: ", this.name);
        for (int i = 0; i < frontConnections.size(); i++) {
            System.out.printf(" [ %-2s = Peso: %-10s] ", frontConnections.get(i).getNeuron().getName(), formatDouble.format(frontConnections.get(i).getWeight()));
        }
        if (frontConnections.isEmpty()) {
            System.out.printf("%40s  %10s", "Sin conexiones hacia adelante", " ");
        }
        System.out.printf("|  \n| %-15s Conexiones <--: ", " ");
        for (int i = 0; i < backConnections.size(); i++) {
            System.out.printf(" [ %-2s = Peso: %-10s] ", backConnections.get(i).getNeuron().getName(), formatDouble.format(backConnections.get(i).getWeight()));
        }
        if (backConnections.isEmpty()) {
            System.out.printf(" %36s %15s \n", "Sin conexiones hacia atras", "|");
        } else {
            System.out.print("|\n");
        }
    }
}
