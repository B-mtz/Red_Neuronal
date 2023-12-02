import java.util.ArrayList;

public class Execute {
    public static void main(String[] args) {
        ArrayList<Integer> topologia = new ArrayList<>();
        ArrayList<Double> entradas = new ArrayList<>();
        ArrayList<Double> targets = new ArrayList<>();

        //------------Topologia------------
        topologia.add(2);
        topologia.add(2);
        topologia.add(2);

        //------------Entradas------------
        entradas.add(0.1);
        entradas.add(0.5);

        //------------Salidas------------
        targets.add(0.05);
        targets.add(0.95);

        NeuronalNetwork network = new NeuronalNetwork(topologia,entradas,targets);
    }
}
