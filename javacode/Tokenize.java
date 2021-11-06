package dhl;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.Tree;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class Tokenize {
    public static BufferedReader bufReader(String fileName) throws IOException {
        FileInputStream fstream = new FileInputStream(fileName);
        return new BufferedReader(new InputStreamReader(fstream, "UTF8"));
    }

    public static BufferedWriter bufWriter(String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        return new BufferedWriter(new OutputStreamWriter(fos, "UTF8"));
    }

    public static void tokenizeEveryLine() throws Exception {
        String filename = "rinante-data/semeval14/laptops/laptops_train_texts.txt";
        String dstFile = "rinante-data/semeval14/laptops/laptops_train_texts_tok_pos.txt";
        boolean toLower = true;
        boolean includeSpans = true;
//        boolean toLower = false;
        String optionsStr = "untokenizable=noneKeep";
//        String optionsStr = "untokenizable=noneKeep,ptb3Escaping=false";

        CoreLabelTokenFactory tf = new CoreLabelTokenFactory();
        BufferedReader reader = bufReader(filename);
        BufferedWriter writer = bufWriter(dstFile);
        String line = null;
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            PTBTokenizer ptbt = new PTBTokenizer<CoreLabel>(new StringReader(line), tf, optionsStr);
            List<CoreLabel> tokens = ptbt.tokenize();
            boolean first = true;
            for (CoreLabel cl : tokens) {
                String w = toLower ? cl.value().toLowerCase() : cl.value();
                if (first) {
                    writer.write(w);
                    first = false;
                } else {
                    writer.write(String.format(" %s", w));
                }
            }
            writer.write("\n");

            if (includeSpans) {
                first = true;
                for (CoreLabel cl : tokens) {
                    if (first) {
                        writer.write(String.format("%d %d", cl.beginPosition(), cl.endPosition()));
                        first = false;
                    } else {
                        writer.write(String.format(" %d %d", cl.beginPosition(), cl.endPosition()));
                    }
                }
                writer.write("\n");
            }

            ++cnt;
            if (cnt % 1000000 == 0) {
                System.out.println(cnt);
            }
        }
        reader.close();
        writer.close();
    }

    public static void main(String[] args) throws Exception {
        tokenizeEveryLine();
    }
}
