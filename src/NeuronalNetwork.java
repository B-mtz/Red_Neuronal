import Model.Connection;
import Model.Neuron;

import java.util.ArrayList;
import java.util.LinkedList;

public class NeuronalNetwork {
    private ArrayList<Integer> topology;
    private ArrayList<Double> inputs, targets;
    private LinkedList<Neuron> neurons;
    private Double totalError;
    private static final int MAX_RANGE= 10;

    public NeuronalNetwork( ArrayList<Integer> topology, ArrayList<Double> inputs,ArrayList<Double> targets){
        this.topology = topology;
        this.inputs = inputs;
        this.targets = targets;
        this.neurons = new LinkedList<>();
        this.totalError = 0.0;
        //neuronGenerator();
        test();
        printNeurons();
        //connectionGenerator();
        printConnections();
        feedForward();
        printNeurons();
    }

    //Clase para probar que funcionen: progrpacion hacia adelante y atras, error total, error conmutado, nuevos valores de w y b
    private  void test(){
        Neuron i1 = new Neuron("i1",0,0.1,0);
        Neuron i2 = new Neuron("i2",0,0.5,0);
        Neuron h1 = new Neuron("h1",1,0.25);
        Neuron h2 = new Neuron("h2",1,0.25);
        Neuron o1 = new Neuron("o1",2,0.05,0.35);
        Neuron o2 = new Neuron("o2",2,0.95,0.35);
        //Conexiones hacia adelante:
        i1.getFrontConnections().add(new Connection(h1,0.1));
        i1.getFrontConnections().add(new Connection(h2,0.2));
        i2.getFrontConnections().add(new Connection(h1,0.3));
        i2.getFrontConnections().add(new Connection(h2,0.4));
        h1.getFrontConnections().add(new Connection(o1,0.5));
        h1.getFrontConnections().add(new Connection(o2,0.7));
        h2.getFrontConnections().add(new Connection(o1,0.6));
        h2.getFrontConnections().add(new Connection(o2,0.8));
        //Conexiones hacia atras
        o1.getBackConnections().add(new Connection(h1,0.5));
        o1.getBackConnections().add(new Connection(h2,0.6));
        o2.getBackConnections().add(new Connection(h1,0.7));
        o2.getBackConnections().add(new Connection(h2,0.8));
        h1.getBackConnections().add(new Connection(i1,0.1));
        h1.getBackConnections().add(new Connection(i2,0.3));
        h2.getBackConnections().add(new Connection(i1,0.2));
        h2.getBackConnections().add(new Connection(i2,0.4));
        //Neuronas
        neurons.add(i1);
        neurons.add(i2);
        neurons.add(h1);
        neurons.add(h2);
        neurons.add(o1);
        neurons.add(o2);

    }

    //Crear las neuronas con bias aletorios
    private void neuronGenerator() {
        int contInput = 1, contOutput=1, contW = 1;
        for (int i = 0; i < topology.size(); i++) {
            int cantidad = topology.get(i);
            for (int j = 0; j < cantidad; j++) {
                if (i == 0) {//Neuronas de la capa 0 = capa de inputs
                    String name = "i"+contInput;
                    Neuron neurona = new Neuron(name,i,inputs.get(j), 0);
                    neurons.add(neurona);
                    contInput++;
                } else if (i == topology.size() - 1) {//Neuronas de la capa de outputs o targets
                    String name = "o"+contOutput;
                    Neuron neurona = new Neuron(name,i,targets.get(j), randomValue());
                    neurons.add(neurona);
                    contOutput++;
                } else {//Neuronas de las capas ocultas
                    String name = "h"+contW;
                    Neuron neurona = new Neuron(name,i, randomValue());
                    neurons.add(neurona);
                    contW++;
                }
            }
        }
    }

    // Crea las conexiones entre las neuronas con pesos aleatorios
    private void connectionGenerator() {
        int actualLayer = 0;
        for (int i = 0; i < neurons.size(); i++) {//Recorre las neuronas apartir de la primera
            if (actualLayer != neurons.get(i).getLayer()) {//Valida que las neuronas pertenezcan a las mismas capas
                actualLayer++;
            }
            for (int j = 1; j < neurons.size(); j++) {//Recorre las neuronas apartir de la segunda
                if (neurons.get(j).getLayer() == actualLayer + 1) {//Valida que las neuronas pertenezcan a otra capa
                    //Se agregan conexiones traseras
                    Connection backConn = new Connection(neurons.get(i), randomValue());
                    neurons.get(j).getBackConnections().add(backConn);
                    //Se agregan conexiones delanteras
                    Connection frontConn = new Connection(neurons.get(j), randomValue());
                    neurons.get(i).getFrontConnections().add(frontConn);
                }
            }
        }
    }
    //Propagación hacia adelante
    private void feedForward() {
        for (Neuron n : neurons) {
            if (n.getLayer() != 0) {
                double sumh = 0;
                if (n.getLayer() == 1) {//Verifica si es una neurona de entrada
                    for (Connection c : n.getBackConnections()) {
                        sumh += c.getNeuron().getValue()* c.getWeight();// sumh = i1*W1 + i2*w3
                    }
                    sumh += n.getBias(); //sumh + bias
                    n.setSumTotal(sumh);
                    n.setOutput(activatioFunction(sumh));
                } else {//Se calcula la suma ponderada de las neuronas de la capa oculta y de salida
                    for (Connection c : n.getBackConnections()) {
                        sumh += c.getNeuron().getOutput() * c.getWeight();// sumh = i1*W1 + i2*w3
                    }
                    sumh += n.getBias();
                    n.setSumTotal(sumh);//sumh + bias
                    n.setOutput(activatioFunction(sumh));
                }
            }
        }
        getTotalError();
    }

    //Calcular Output de las neuronas
    private double activatioFunction(double sumh) {
        double total = 1 / (1 + Math.exp(-sumh));
        return total;
    }

    //Calcular el error Total
    private void getTotalError() {
        double total = 0;
        for (Neuron n : neurons) {
            if (n.getLayer() == topology.size() - 1) {//Se filtra para la ultima capa
                total += (Math.pow(n.getValue() - n.getOutput(), 2)) / 2; // 1/2 * (target-output)^2
            }
        }
        totalError = total;
        System.out.println("Error total :" + totalError);
    }

    //Genera valores aleatorios con un ranto 0.1 a MAX_RANGE = 10
    private double randomValue(){
        return Math.random()*(MAX_RANGE-0.1)+0.1;
    }
    //Imprime las neuronas
    private void printNeurons() {
        System.out.println("\nCantidad de neuronas : " + neurons.size());
        System.out.println("--------------------------------------------------------------------NEURONAS--------------------------------------------------------------------");
        for (Neuron n : neurons) {
            System.out.println(n.toString());
        }
        System.out.println("------------------------------------------------------------------------------------------------------------------------------------------------");
    }
    //Imprime las conexiones
    private void printConnections() {
        System.out.println("\n--------------------------------------CONEXIONES---------------------------------------");
        for (Neuron n : neurons) {
            n.printConnections();
        }
        System.out.println("---------------------------------------------------------------------------------------");
    }

}
