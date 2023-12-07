package test1;

import data.CSVReader;

import java.util.ArrayList;
import java.util.List;

public class Execute {
    private static final String filePath = "src/data/Iris_Intercalado.csv";
    private static final int epoca = 100;
    public static void main(String[] args) {
        //================Se lee el archivo CSV================
        CSVReader csvReader = new CSVReader();
        Execute execute = new Execute();
        List<String[]> data = csvReader.readCSV(filePath);

        //================Se ejecuta la red Neuronal================
        ArrayList<Integer> topology = new ArrayList<>();
        topology.add(4);
        topology.add(6);
        topology.add(6);
        topology.add(2);

        NeuronalNetwork network = new NeuronalNetwork(topology, data);
        System.out.println("RED NEURONAL: Entrenando...");
        for (int i = 0; i < epoca; i++) {
            network.executeTraining();
        }
        System.out.println("Error Final : % "+ network.totalError*100);
        network.executeTest();

    }
}