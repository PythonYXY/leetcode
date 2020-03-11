import java.io.*;
import java.util.*;

public class Script {
    public static void main(String[] args) throws Exception {
        File folder = new File(".");
        File[] listOfFiles = folder.listFiles();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isDirectory()) {
                File dir = listOfFiles[i];
                System.out.println("* [" + dir.getName() + "]()");
                File[] listOfDir = dir.listFiles();
                for (File file: listOfDir) {
                    System.out.println("\t* [" + file.getName().replace(".md", " ").trim() + "](" + dir.getName() + "/" + file.getName() + ")");
                }
            }
        }
    }

}