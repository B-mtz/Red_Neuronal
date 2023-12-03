package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {
    private static final String filePath = "src/data/Iris.csv";
    public static List<String[]> readCSV() {
        List<String[]> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                lines.add(values);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lines;
    }

    public void printData() {
        List<String[]> data = readCSV();
        // Imprimir los datos
        for (String[] row : data) {
            for (String value : row) {
                System.out.print(value + "");
            }
            System.out.println();
        }
    }
}

