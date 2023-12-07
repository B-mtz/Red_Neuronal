package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {
    List<String[]> lines;
    public List<String[]> readCSV(String filePath) {
        lines = new ArrayList<>();
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
        // Imprimir los datos
        for (String[] row : lines) {
            for (String value : row) {
                System.out.print(value + "_");
            }
            System.out.println();
        }
    }
}

