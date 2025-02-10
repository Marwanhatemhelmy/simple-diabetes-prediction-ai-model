// this class was created with the help of ai
public class ZScoreNormalization {
    public static void normalize(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        
        double[] means = new double[cols];
        double[] stdDevs = new double[cols];

        // Calculate mean for each feature
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            means[j] = sum / rows;
        }

        // Calculate standard deviation for each feature
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += Math.pow(data[i][j] - means[j], 2);
            }
            stdDevs[j] = Math.sqrt(sum / rows); // Population standard deviation
        }

        // Normalize each value using Z-score formula
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (stdDevs[j] != 0) { // Avoid division by zero
                    data[i][j] = (data[i][j] - means[j]) / stdDevs[j];
                } else {
                    data[i][j] = 0; // If standard deviation is zero, set it to 0
                }
            }
        }
    }
}