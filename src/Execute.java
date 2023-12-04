import data.CSVReader;

import java.util.ArrayList;
import java.util.List;

public class Execute {
    private static final String filePath = "src/data/Iris_Intercalado.csv";

    public static void main(String[] args) {
        //================Se lee el archivo CSV================
        CSVReader csvReader = new CSVReader();
        List<String[]> data = csvReader.readCSV(filePath);

        //================Se ejecuta la red Neuronal================
        ArrayList<Integer> topology = new ArrayList<>();

        //------------Topologia------------
        topology.add(4);
        topology.add(6);
        topology.add(2);
        //------------Inicializar la red neuronal------------
        ArrayList<Double> inputs = new ArrayList<>();
        inputs.add(0.0);
        inputs.add(0.0);
        inputs.add(0.0);
        inputs.add(0.0);
        ArrayList<Double> targets = new ArrayList<>();
        targets.add(0.0);
        targets.add(0.0);

        NeuronalNetwork network = new NeuronalNetwork(topology, inputs, targets);

        int count = 0;
        System.out.println("RED NEURONAL: Entrenando...");
        for (String[] row : data) {
            inputs = new ArrayList<>();
            targets = new ArrayList<>();
            for (int i = 1; i < row.length; i++) {
                if (row[i].equalsIgnoreCase("Iris-setosa")) {
                    targets.add(0.0);
                    targets.add(0.0);
                } else if (row[i].equalsIgnoreCase("Iris-versicolor")) {
                    targets.add(0.0);
                    targets.add(1.0);
                } else if (row[i].equalsIgnoreCase("Iris-virginica")) {
                    targets.add(1.0);
                    targets.add(1.0);
                } else {
                    inputs.add(Double.valueOf(row[i]));
                }
            }
            if (count<130){
                network.replaceInputs(inputs);
                network.replaceTargets(targets);
                network.executeTraining();//Ejecuta el entrenamiento
            }else{
                if (count == 130){
                    System.out.println("RED NEURONAL: Testeando datos...");
                }
                network.replaceInputs(inputs);
                network.replaceTargets(targets);
                network.executeTest();//Ejecuta el test
            }
            count++;
        }
    }
}
