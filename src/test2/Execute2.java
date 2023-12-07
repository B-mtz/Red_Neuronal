package test2;

import data.CSVReader;
import test1.NeuronalNetwork;

import java.util.ArrayList;
import java.util.List;

public class Execute2 {
    private static final String filePath = "src/data/Cancer.csv";
    private static final int epoca = 500;
    public static void main(String[] args) {
        //================Se lee el archivo CSV================
        CSVReader csvReader = new CSVReader();
        Execute2 execute = new Execute2();
        List<String[]> data = csvReader.readCSV(filePath);

        //================Se ejecuta la red Neuronal================
        ArrayList<Integer> topology = new ArrayList<>();
        topology.add(30);
        topology.add(35);
        topology.add(2);

        NeuronalNetwork2 network = new NeuronalNetwork2(topology, data);
        System.out.println("RED NEURONAL: Entrenando...");
        for (int i = 0; i < epoca; i++) {
            network.executeTraining();
        }
        System.out.println("Error Final : % "+ network.totalError*100);
        network.executeTest();

    }
}