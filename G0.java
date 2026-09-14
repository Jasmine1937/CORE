import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

/**
 * Standalone G0 implementation of the manuscript's GP search.
 *
 * Representation: prefix integer trees. Terminals are 0..(nFeatures-1),
 * while ADD/SUB/MUL are negative integer codes. All variants share
 * initialization, variation and tie-aware AUC implementation.
 */
public final class G0 {
    private static final int ADD = -1;
    private static final int SUB = -2;
    private static final int MUL = -3;

    private final String variant;
    private final Options opt;
    private final Random random;

    private double[][] features;
    private int[] labels;
    private int featureCount;

    private int[][] population;
    private double[] auc;
    private double[] selectionFitness;
    private int[] treeSize;
    private int[] rank;
    private double[] crowding;

    private PrintWriter candidateWriter;
    private final Map<String, ArchivedCandidate> archive = new LinkedHashMap<>();
    private final Set<String> allCanonicalSeen = new HashSet<>();
    private final Map<String, SeenMetadata> seenMetadata = new HashMap<>();
    private long rawArchiveObservations = 0L;
    private long maxUniqueBeforeBudget = 0L;

    private G0(String variant, Options opt) {
        this.variant = variant;
        this.opt = opt;
        this.random = new Random(opt.seed);
    }

    public static void main(String[] args) throws Exception {
        String variant = "G0";
        Locale.setDefault(Locale.ROOT);
        Options opt = Options.parse(args);
        G0 engine = new G0(variant, opt);
        engine.run();
    }

    private void run() throws Exception {
        loadData(opt.dataPath);
        Path parent = opt.candidatePath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(
                opt.candidatePath, StandardCharsets.UTF_8))) {
            candidateWriter = out;
            writeCandidateHeader();
            initializePopulation();
            evaluatePopulation();
            if (variant.equals("G2")) {
                computeParetoRankAndCrowding();
            }
            reportAndArchive(0);

            for (int generation = 1; generation < opt.generations; generation++) {
                int[][] next = new int[opt.populationSize][];
                for (int i = 0; i < opt.populationSize; i++) {
                    if (random.nextDouble() < opt.crossoverProbability) {
                        next[i] = crossover(population[tournament()], population[tournament()]);
                    } else {
                        next[i] = mutate(population[tournament()]);
                    }
                }
                population = next; // Full generational replacement without elitism.
                evaluatePopulation();
                if (variant.equals("G2")) {
                    computeParetoRankAndCrowding();
                }
                reportAndArchive(generation);
            }
            writeFinalArchive();
        }
    }

    private void loadData(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IOException("Empty GP data file: " + path);
            }
            StringTokenizer head = new StringTokenizer(header);
            if (head.countTokens() < 5) {
                throw new IOException("Invalid GP header (expected 5 values): " + header);
            }
            featureCount = Integer.parseInt(head.nextToken());
            int randomConstants = Integer.parseInt(head.nextToken());
            head.nextToken(); // Random-constant minimum field; unused when the count is zero.
            head.nextToken(); // Random-constant maximum field; unused when the count is zero.
            int rows = Integer.parseInt(head.nextToken());
            if (randomConstants != 0) {
                throw new IOException("Random-constant count in the GP header must be 0");
            }
            if (featureCount <= 0 || rows <= 0) {
                throw new IOException("Feature and row counts must be positive");
            }

            features = new double[rows][featureCount];
            labels = new int[rows];
            for (int i = 0; i < rows; i++) {
                String line = reader.readLine();
                if (line == null) {
                    throw new IOException("Unexpected EOF at data row " + (i + 1));
                }
                StringTokenizer tokens = new StringTokenizer(line);
                if (tokens.countTokens() != featureCount + 1) {
                    throw new IOException("Row " + (i + 1) + " has " + tokens.countTokens()
                            + " values; expected " + (featureCount + 1));
                }
                for (int j = 0; j < featureCount; j++) {
                    features[i][j] = Double.parseDouble(tokens.nextToken());
                    if (!Double.isFinite(features[i][j])) {
                        throw new IOException("Non-finite feature at row " + (i + 1) + ", column " + j);
                    }
                }
                double label = Double.parseDouble(tokens.nextToken());
                if (label != 0.0 && label != 1.0) {
                    throw new IOException("Label must be 0/1 at row " + (i + 1));
                }
                labels[i] = (int) label;
            }
            if (reader.readLine() != null) {
                throw new IOException("Data file contains more rows than declared in its header");
            }
        }
    }

    private void initializePopulation() {
        population = new int[opt.populationSize][];
        for (int i = 0; i < opt.populationSize; i++) {
            IntBuffer buffer = new IntBuffer(Math.min(opt.maxLength, 64));
            grow(buffer, opt.initialDepth);
            population[i] = buffer.toArray();
        }
    }

    private void grow(IntBuffer buffer, int depth) {
        if (buffer.size() >= opt.maxLength) {
            throw new IllegalStateException("Initial tree exceeded max length " + opt.maxLength);
        }
        if (depth > 0 && random.nextDouble() < 0.5) {
            buffer.add(randomFunction());
            grow(buffer, depth - 1);
            grow(buffer, depth - 1);
        } else {
            buffer.add(random.nextInt(featureCount));
        }
    }

    private int randomFunction() {
        int choice = random.nextInt(3);
        return choice == 0 ? ADD : (choice == 1 ? SUB : MUL);
    }

    private void evaluatePopulation() {
        int n = population.length;
        auc = new double[n];
        selectionFitness = new double[n];
        treeSize = new int[n];
        rank = new int[n];
        crowding = new double[n];

        for (int i = 0; i < n; i++) {
            int[] program = population[i];
            treeSize[i] = program.length;
            double[] scores = new double[features.length];
            double[] evaluationStack = new double[program.length];
            for (int row = 0; row < features.length; row++) {
                double value = evaluate(program, features[row], evaluationStack);
                if (!Double.isFinite(value)) {
                    value = 0.0;
                }
                scores[row] = value;
            }
            auc[i] = tieAwareAuc(scores, labels);
            if (variant.equals("G1")) {
                double penalty = Math.log1p(treeSize[i]) / Math.log1p(opt.treeNorm);
                selectionFitness[i] = auc[i] - opt.lambda * penalty;
            } else {
                selectionFitness[i] = auc[i];
            }
        }
    }

    private double evaluate(int[] program, double[] row, double[] stack) {
        int size = 0;
        for (int i = program.length - 1; i >= 0; i--) {
            int token = program[i];
            if (token >= 0) {
                stack[size++] = row[token];
            } else {
                if (size < 2) throw new IllegalStateException("Malformed prefix program");
                double left = stack[--size];
                double right = stack[--size];
                double value;
                if (token == ADD) value = left + right;
                else if (token == SUB) value = left - right;
                else if (token == MUL) value = left * right;
                else throw new IllegalStateException("Unknown function token: " + token);
                stack[size++] = value;
            }
        }
        if (size != 1) throw new IllegalStateException("Malformed prefix program");
        return stack[0];
    }

    /** AUC with half credit for tied positive-negative score pairs. */
    static double tieAwareAuc(double[] scores, int[] y) {
        if (scores.length != y.length || scores.length == 0) {
            throw new IllegalArgumentException("Scores and labels must have equal non-zero length");
        }
        int positive = 0;
        for (int label : y) {
            if (label == 1) positive++;
            else if (label != 0) throw new IllegalArgumentException("Labels must be 0/1");
        }
        int negative = y.length - positive;
        if (positive == 0 || negative == 0) return 0.5;

        // Sorting the two classes separately avoids allocating boxed row indices.
        // AUC is the probability that a positive score exceeds a negative score,
        // with a half credit for exact ties.
        double[] positiveScores = new double[positive];
        double[] negativeScores = new double[negative];
        int pIndex = 0;
        int nIndex = 0;
        for (int i = 0; i < scores.length; i++) {
            if (y[i] == 1) positiveScores[pIndex++] = scores[i];
            else negativeScores[nIndex++] = scores[i];
        }
        Arrays.sort(positiveScores);
        Arrays.sort(negativeScores);
        double favorablePairs = 0.0;
        int p = 0;
        int n = 0;
        while (p < positiveScores.length) {
            double score = positiveScores[p];
            int pEnd = p + 1;
            while (pEnd < positiveScores.length && positiveScores[pEnd] == score) pEnd++;
            while (n < negativeScores.length && negativeScores[n] < score) n++;
            int tiedEnd = n;
            while (tiedEnd < negativeScores.length && negativeScores[tiedEnd] == score) tiedEnd++;
            favorablePairs += (pEnd - p) * (n + 0.5 * (tiedEnd - n));
            p = pEnd;
        }
        double aucValue = favorablePairs / ((double) positive * negative);
        return Math.max(0.0, Math.min(1.0, aucValue));
    }

    private int tournament() {
        int best = random.nextInt(opt.populationSize);
        for (int i = 1; i < opt.tournamentSize; i++) {
            int competitor = random.nextInt(opt.populationSize);
            if (better(competitor, best)) best = competitor;
        }
        return best;
    }

    private boolean better(int a, int b) {
        if (variant.equals("G2")) {
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            if (Double.compare(crowding[a], crowding[b]) != 0) return crowding[a] > crowding[b];
            if (Double.compare(auc[a], auc[b]) != 0) return auc[a] > auc[b];
            return treeSize[a] < treeSize[b];
        }
        double fa = variant.equals("G1") ? selectionFitness[a] : auc[a];
        double fb = variant.equals("G1") ? selectionFitness[b] : auc[b];
        // G0 compares AUC; G1 compares penalized scalar fitness.
        // Scalar-fitness ties preserve the current tournament winner.
        return fa > fb;
    }

    private int[] crossover(int[] parent1, int[] parent2) {
        int start1 = random.nextInt(parent1.length);
        int end1 = subtreeEnd(parent1, start1);
        int start2 = random.nextInt(parent2.length);
        int end2 = subtreeEnd(parent2, start2);
        int newLength = start1 + (end2 - start2) + (parent1.length - end1);
        if (newLength >= opt.maxLength) return parent1;

        int[] child = new int[newLength];
        System.arraycopy(parent1, 0, child, 0, start1);
        System.arraycopy(parent2, start2, child, start1, end2 - start2);
        System.arraycopy(parent1, end1, child, start1 + end2 - start2, parent1.length - end1);
        return child;
    }

    private int[] mutate(int[] parent) {
        int[] child = parent.clone();
        int site = random.nextInt(child.length);
        child[site] = child[site] >= 0 ? random.nextInt(featureCount) : randomFunction();
        return child;
    }

    private int subtreeEnd(int[] program, int start) {
        int pending = 1;
        int position = start;
        while (pending > 0) {
            if (position >= program.length) throw new IllegalStateException("Malformed prefix program");
            int token = program[position++];
            pending--;
            if (token < 0) pending += 2;
        }
        return position;
    }

    private void reportAndArchive(int generation) {
        int rawBest = argMax(auc);
        double average = Arrays.stream(auc).average().orElse(Double.NaN);
        System.out.println("Generation " + generation + ": Avg Fitness = " + average
                + ", Best = " + auc[rawBest]);
        System.out.println("Best Individual: " + expression(population[rawBest]));

        if (variant.equals("G0")) {
            addToArchive(generation, rawBest, "native_auc_generation_representative");
            truncateScalarArchive(false);
        } else if (variant.equals("G1")) {
            int penalizedBest = argMax(selectionFitness);
            addToArchive(generation, penalizedBest,
                    "native_penalized_fitness_generation_representative");
            truncateScalarArchive(true);
        } else {
            for (int i = 0; i < population.length; i++) {
                if (rank[i] == 0) {
                    addToArchive(generation, i, "bounded_cross_generation_pareto_archive");
                }
            }
            recomputeAndTruncateParetoArchive();
        }
    }

    private int argMax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private void writeCandidateHeader() {
        candidateWriter.println("variant\tmode\treplicate\tseed\twindow\tgeneration\tarchive_reason"
                + "\ttrain_auc\tselection_fitness\ttree_size\tpareto_rank\tcrowding\texpression"
                + "\tfirst_seen_generation\tlast_seen_generation\toccurrence_count"
                + "\tevolution_auc\tnative_fitness\tdepth\tnode_count\toperator_count"
                + "\tvariable_count\tfeature_set\tcanonical_expression\tarchive_rule"
                + "\tarchive_budget\tn_raw_archive_observations\tn_unique_before_budget"
                + "\tcrowding_distance");
    }

    private void writeFinalArchive() {
        List<ArchivedCandidate> finalArchive = new ArrayList<>(archive.values());
        if (variant.equals("G2")) {
            finalArchive.sort(paretoArchiveComparator());
        } else {
            finalArchive.sort(Comparator
                    .comparingInt((ArchivedCandidate item) -> item.firstSeenGeneration)
                    .thenComparing(item -> item.canonicalExpression));
        }
        for (ArchivedCandidate item : finalArchive) {
            writeCandidate(item);
        }
        candidateWriter.flush();
        System.out.println("Frozen archive: raw observations=" + rawArchiveObservations
                + ", unique ever seen=" + allCanonicalSeen.size()
                + ", retained=" + finalArchive.size()
                + ", budget=" + opt.archiveBudget);
    }

    private void writeCandidate(ArchivedCandidate item) {
        String fitness = variant.equals("G2") ? "" : Double.toString(item.nativeFitness);
        String paretoRank = variant.equals("G2") ? Integer.toString(item.paretoRank) : "";
        String crowd = variant.equals("G2") ? Double.toString(item.crowdingDistance) : "";
        candidateWriter.printf(Locale.ROOT,
                "%s\trevision\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\t%s\t%s\t%s"
                        + "\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s"
                        + "\t%d\t%d\t%d\t%s%n",
                variant, opt.replicate, opt.seed, opt.window, item.selectedGeneration,
                item.archiveReason, Double.toString(item.evolutionAuc), fitness, item.treeSize,
                paretoRank, crowd, item.rawExpression,
                item.firstSeenGeneration, item.lastSeenGeneration, item.occurrenceCount,
                Double.toString(item.evolutionAuc), fitness, item.depth, item.treeSize,
                item.operatorCount, item.features.size(), featureSetText(item.features),
                item.canonicalExpression, item.archiveReason, opt.archiveBudget,
                rawArchiveObservations, allCanonicalSeen.size(), crowd);
    }

    private void addToArchive(int generation, int individual, String reason) {
        rawArchiveObservations++;
        ProgramDescription description = describe(population[individual]);
        allCanonicalSeen.add(description.canonicalExpression);
        SeenMetadata seen = seenMetadata.get(description.canonicalExpression);
        if (seen == null) {
            seen = new SeenMetadata(generation);
            seenMetadata.put(description.canonicalExpression, seen);
        } else {
            seen.lastSeenGeneration = generation;
            seen.occurrenceCount++;
        }
        ArchivedCandidate existing = archive.get(description.canonicalExpression);
        if (existing == null) {
            ArchivedCandidate item = new ArchivedCandidate(
                    population[individual].clone(), description.rawExpression,
                    description.canonicalExpression, auc[individual], selectionFitness[individual],
                    treeSize[individual], description.depth, description.operatorCount,
                    description.features, generation, reason);
            item.firstSeenGeneration = seen.firstSeenGeneration;
            item.lastSeenGeneration = seen.lastSeenGeneration;
            item.occurrenceCount = seen.occurrenceCount;
            archive.put(item.canonicalExpression, item);
        } else {
            existing.lastSeenGeneration = seen.lastSeenGeneration;
            existing.occurrenceCount = seen.occurrenceCount;
            if (betterArchivedObservation(individual, description, existing)) {
                existing.program = population[individual].clone();
                existing.rawExpression = description.rawExpression;
                existing.evolutionAuc = auc[individual];
                existing.nativeFitness = selectionFitness[individual];
                existing.treeSize = treeSize[individual];
                existing.depth = description.depth;
                existing.operatorCount = description.operatorCount;
                existing.features = new TreeSet<>(description.features);
                existing.selectedGeneration = generation;
            }
        }
        maxUniqueBeforeBudget = Math.max(maxUniqueBeforeBudget, archive.size());
    }

    private boolean betterArchivedObservation(
            int individual, ProgramDescription description, ArchivedCandidate existing) {
        double newPrimary = variant.equals("G1") ? selectionFitness[individual] : auc[individual];
        double oldPrimary = variant.equals("G1") ? existing.nativeFitness : existing.evolutionAuc;
        int primaryComparison = Double.compare(newPrimary, oldPrimary);
        if (primaryComparison != 0) return primaryComparison > 0;
        int aucComparison = Double.compare(auc[individual], existing.evolutionAuc);
        if (aucComparison != 0) return aucComparison > 0;
        if (treeSize[individual] != existing.treeSize) return treeSize[individual] < existing.treeSize;
        if (description.depth != existing.depth) return description.depth < existing.depth;
        return description.rawExpression.compareTo(existing.rawExpression) < 0;
    }

    private void truncateScalarArchive(boolean penalized) {
        if (archive.size() <= opt.archiveBudget) return;
        List<ArchivedCandidate> ordered = new ArrayList<>(archive.values());
        Comparator<ArchivedCandidate> comparator = Comparator
                .comparingDouble((ArchivedCandidate item) ->
                        penalized ? item.nativeFitness : item.evolutionAuc).reversed()
                .thenComparing(Comparator.comparingDouble(
                        (ArchivedCandidate item) -> item.evolutionAuc).reversed())
                .thenComparingInt(item -> item.treeSize)
                .thenComparingInt(item -> item.depth)
                .thenComparingInt(item -> item.firstSeenGeneration)
                .thenComparing(item -> item.canonicalExpression);
        ordered.sort(comparator);
        replaceArchive(ordered.subList(0, opt.archiveBudget));
    }

    private void recomputeAndTruncateParetoArchive() {
        List<ArchivedCandidate> items = new ArrayList<>(archive.values());
        for (ArchivedCandidate item : items) {
            item.paretoRank = -1;
            item.crowdingDistance = 0.0;
        }
        int n = items.size();
        int[] dominationCount = new int[n];
        List<List<Integer>> dominates = new ArrayList<>(n);
        for (int i = 0; i < n; i++) dominates.add(new ArrayList<>());
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                if (dominates(items.get(p), items.get(q))) {
                    dominates.get(p).add(q);
                    dominationCount[q]++;
                } else if (dominates(items.get(q), items.get(p))) {
                    dominates.get(q).add(p);
                    dominationCount[p]++;
                }
            }
        }
        List<Integer> current = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (dominationCount[i] == 0) current.add(i);
        }
        int currentRank = 0;
        while (!current.isEmpty()) {
            for (int index : current) items.get(index).paretoRank = currentRank;
            addArchiveCrowding(items, current, true);
            addArchiveCrowding(items, current, false);
            List<Integer> next = new ArrayList<>();
            for (int p : current) {
                for (int q : dominates.get(p)) {
                    dominationCount[q]--;
                    if (dominationCount[q] == 0) next.add(q);
                }
            }
            current = next;
            currentRank++;
        }
        items.sort(paretoArchiveComparator());
        int retain = Math.min(opt.archiveBudget, items.size());
        replaceArchive(items.subList(0, retain));
    }

    private boolean dominates(ArchivedCandidate p, ArchivedCandidate q) {
        boolean noWorse = p.evolutionAuc >= q.evolutionAuc && p.treeSize <= q.treeSize;
        boolean strictlyBetter = p.evolutionAuc > q.evolutionAuc || p.treeSize < q.treeSize;
        return noWorse && strictlyBetter;
    }

    private void addArchiveCrowding(
            List<ArchivedCandidate> items, List<Integer> front, boolean errorObjective) {
        if (front.isEmpty()) return;
        if (front.size() <= 2) {
            for (int index : front) items.get(index).crowdingDistance = Double.POSITIVE_INFINITY;
            return;
        }
        Integer[] order = front.toArray(new Integer[0]);
        Arrays.sort(order, Comparator.comparingDouble(index ->
                archiveObjective(items.get(index), errorObjective)));
        items.get(order[0]).crowdingDistance = Double.POSITIVE_INFINITY;
        items.get(order[order.length - 1]).crowdingDistance = Double.POSITIVE_INFINITY;
        double minimum = archiveObjective(items.get(order[0]), errorObjective);
        double maximum = archiveObjective(items.get(order[order.length - 1]), errorObjective);
        if (!(maximum > minimum)) return;
        for (int i = 1; i < order.length - 1; i++) {
            ArchivedCandidate item = items.get(order[i]);
            if (!Double.isInfinite(item.crowdingDistance)) {
                double previous = archiveObjective(items.get(order[i - 1]), errorObjective);
                double next = archiveObjective(items.get(order[i + 1]), errorObjective);
                item.crowdingDistance += (next - previous) / (maximum - minimum);
            }
        }
    }

    private double archiveObjective(ArchivedCandidate item, boolean errorObjective) {
        return errorObjective ? 1.0 - item.evolutionAuc : item.treeSize;
    }

    private Comparator<ArchivedCandidate> paretoArchiveComparator() {
        return Comparator.comparingInt((ArchivedCandidate item) -> item.paretoRank)
                .thenComparing(Comparator.comparingDouble(
                        (ArchivedCandidate item) -> item.crowdingDistance).reversed())
                .thenComparing(Comparator.comparingDouble(
                        (ArchivedCandidate item) -> item.evolutionAuc).reversed())
                .thenComparingInt(item -> item.treeSize)
                .thenComparingInt(item -> item.depth)
                .thenComparingInt(item -> item.firstSeenGeneration)
                .thenComparing(item -> item.canonicalExpression);
    }

    private void replaceArchive(List<ArchivedCandidate> retained) {
        archive.clear();
        for (ArchivedCandidate item : retained) archive.put(item.canonicalExpression, item);
    }

    private String featureSetText(Set<Integer> featuresUsed) {
        StringBuilder text = new StringBuilder();
        for (int feature : featuresUsed) {
            if (text.length() > 0) text.append('|');
            text.append('X').append(feature);
        }
        return text.toString();
    }

    private ProgramDescription describe(int[] program) {
        DescriptionCursor cursor = describeAt(program, 0);
        if (cursor.nextPosition != program.length) {
            throw new IllegalStateException("Malformed prefix program");
        }
        return cursor.description;
    }

    private DescriptionCursor describeAt(int[] program, int position) {
        if (position >= program.length) throw new IllegalStateException("Malformed prefix program");
        int token = program[position];
        if (token >= 0) {
            TreeSet<Integer> featuresUsed = new TreeSet<>();
            featuresUsed.add(token);
            ProgramDescription leaf = new ProgramDescription(
                    "X" + token, "X" + token, 1, 0, featuresUsed);
            return new DescriptionCursor(leaf, position + 1);
        }
        DescriptionCursor leftCursor = describeAt(program, position + 1);
        DescriptionCursor rightCursor = describeAt(program, leftCursor.nextPosition);
        ProgramDescription left = leftCursor.description;
        ProgramDescription right = rightCursor.description;
        String operator = token == ADD ? "+" : (token == SUB ? "-" : "*");
        String canonicalLeft = left.canonicalExpression;
        String canonicalRight = right.canonicalExpression;
        if ((token == ADD || token == MUL) && canonicalLeft.compareTo(canonicalRight) > 0) {
            String temporary = canonicalLeft;
            canonicalLeft = canonicalRight;
            canonicalRight = temporary;
        }
        TreeSet<Integer> featuresUsed = new TreeSet<>(left.features);
        featuresUsed.addAll(right.features);
        ProgramDescription node = new ProgramDescription(
                "(" + left.rawExpression + " " + operator + " " + right.rawExpression + ")",
                "(" + canonicalLeft + operator + canonicalRight + ")",
                1 + Math.max(left.depth, right.depth),
                1 + left.operatorCount + right.operatorCount,
                featuresUsed);
        return new DescriptionCursor(node, rightCursor.nextPosition);
    }

    private String expression(int[] program) {
        String[] stack = new String[program.length];
        int size = 0;
        for (int i = program.length - 1; i >= 0; i--) {
            int token = program[i];
            if (token >= 0) {
                stack[size++] = "X" + token;
            } else {
                if (size < 2) throw new IllegalStateException("Malformed prefix program");
                String left = stack[--size];
                String right = stack[--size];
                String operator = token == ADD ? " + " : (token == SUB ? " - " : " * ");
                stack[size++] = "(" + left + operator + right + ")";
            }
        }
        if (size != 1) throw new IllegalStateException("Malformed prefix program");
        return stack[0];
    }

    private void computeParetoRankAndCrowding() {
        int n = population.length;
        Arrays.fill(rank, -1);
        Arrays.fill(crowding, 0.0);
        int[] dominationCount = new int[n];
        List<List<Integer>> dominates = new ArrayList<>(n);
        for (int i = 0; i < n; i++) dominates.add(new ArrayList<>());

        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                if (dominates(p, q)) {
                    dominates.get(p).add(q);
                    dominationCount[q]++;
                } else if (dominates(q, p)) {
                    dominates.get(q).add(p);
                    dominationCount[p]++;
                }
            }
        }

        List<Integer> current = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (dominationCount[i] == 0) current.add(i);
        }
        int currentRank = 0;
        while (!current.isEmpty()) {
            for (int index : current) rank[index] = currentRank;
            addCrowding(current, true);
            addCrowding(current, false);
            List<Integer> next = new ArrayList<>();
            for (int p : current) {
                for (int q : dominates.get(p)) {
                    dominationCount[q]--;
                    if (dominationCount[q] == 0) next.add(q);
                }
            }
            current = next;
            currentRank++;
        }
    }

    private boolean dominates(int p, int q) {
        double errP = 1.0 - auc[p];
        double errQ = 1.0 - auc[q];
        boolean noWorse = errP <= errQ && treeSize[p] <= treeSize[q];
        boolean strictlyBetter = errP < errQ || treeSize[p] < treeSize[q];
        return noWorse && strictlyBetter;
    }

    private void addCrowding(List<Integer> front, boolean errorObjective) {
        if (front.isEmpty()) return;
        if (front.size() <= 2) {
            for (int i : front) crowding[i] = Double.POSITIVE_INFINITY;
            return;
        }
        Integer[] order = front.toArray(new Integer[0]);
        Arrays.sort(order, Comparator.comparingDouble(i -> objective(i, errorObjective)));
        crowding[order[0]] = Double.POSITIVE_INFINITY;
        crowding[order[order.length - 1]] = Double.POSITIVE_INFINITY;
        double min = objective(order[0], errorObjective);
        double max = objective(order[order.length - 1], errorObjective);
        if (!(max > min)) return;
        for (int i = 1; i < order.length - 1; i++) {
            int index = order[i];
            if (!Double.isInfinite(crowding[index])) {
                double previous = objective(order[i - 1], errorObjective);
                double next = objective(order[i + 1], errorObjective);
                crowding[index] += (next - previous) / (max - min);
            }
        }
    }

    private double objective(int individual, boolean errorObjective) {
        return errorObjective ? 1.0 - auc[individual] : treeSize[individual];
    }

    private static final class ProgramDescription {
        final String rawExpression;
        final String canonicalExpression;
        final int depth;
        final int operatorCount;
        final TreeSet<Integer> features;

        ProgramDescription(
                String rawExpression, String canonicalExpression, int depth,
                int operatorCount, Set<Integer> features) {
            this.rawExpression = rawExpression;
            this.canonicalExpression = canonicalExpression;
            this.depth = depth;
            this.operatorCount = operatorCount;
            this.features = new TreeSet<>(features);
        }
    }

    private static final class DescriptionCursor {
        final ProgramDescription description;
        final int nextPosition;

        DescriptionCursor(ProgramDescription description, int nextPosition) {
            this.description = description;
            this.nextPosition = nextPosition;
        }
    }

    private static final class ArchivedCandidate {
        int[] program;
        String rawExpression;
        final String canonicalExpression;
        double evolutionAuc;
        double nativeFitness;
        int treeSize;
        int depth;
        int operatorCount;
        TreeSet<Integer> features;
        int selectedGeneration;
        int firstSeenGeneration;
        int lastSeenGeneration;
        int occurrenceCount;
        int paretoRank = -1;
        double crowdingDistance = 0.0;
        final String archiveReason;

        ArchivedCandidate(
                int[] program, String rawExpression, String canonicalExpression,
                double evolutionAuc, double nativeFitness, int treeSize, int depth,
                int operatorCount, Set<Integer> features, int generation,
                String archiveReason) {
            this.program = program;
            this.rawExpression = rawExpression;
            this.canonicalExpression = canonicalExpression;
            this.evolutionAuc = evolutionAuc;
            this.nativeFitness = nativeFitness;
            this.treeSize = treeSize;
            this.depth = depth;
            this.operatorCount = operatorCount;
            this.features = new TreeSet<>(features);
            this.selectedGeneration = generation;
            this.firstSeenGeneration = generation;
            this.lastSeenGeneration = generation;
            this.occurrenceCount = 1;
            this.archiveReason = archiveReason;
        }
    }

    private static final class SeenMetadata {
        final int firstSeenGeneration;
        int lastSeenGeneration;
        int occurrenceCount;

        SeenMetadata(int generation) {
            this.firstSeenGeneration = generation;
            this.lastSeenGeneration = generation;
            this.occurrenceCount = 1;
        }
    }

    private static final class IntBuffer {
        private int[] data;
        private int size;

        IntBuffer(int capacity) {
            data = new int[Math.max(1, capacity)];
        }

        int size() {
            return size;
        }

        void add(int value) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = value;
        }

        int[] toArray() {
            return Arrays.copyOf(data, size);
        }
    }

    static final class Options {
        Path dataPath;
        Path candidatePath;
        long seed;
        int window;
        int replicate;
        int populationSize = 500;
        int generations = 200;
        int initialDepth = 5;
        int tournamentSize = 2;
        double crossoverProbability = 0.9;
        int maxLength = 10000;
        double lambda = 0.02;
        double treeNorm = 100.0;
        int archiveBudget = 200;

        static Options parse(String[] args) {
            if (args.length == 0 || Arrays.asList(args).contains("--help")) {
                printUsage();
                System.exit(args.length == 0 ? 2 : 0);
            }
            Map<String, String> values = new HashMap<>();
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || i + 1 >= args.length) {
                    throw new IllegalArgumentException("Arguments must be --name value pairs");
                }
                values.put(args[i], args[i + 1]);
            }
            Options opt = new Options();
            opt.dataPath = Paths.get(required(values, "--data"));
            opt.candidatePath = Paths.get(required(values, "--candidates"));
            opt.seed = Long.parseLong(required(values, "--seed"));
            opt.window = Integer.parseInt(required(values, "--window"));
            opt.replicate = Integer.parseInt(required(values, "--replicate"));
            opt.populationSize = integer(values, "--population", opt.populationSize);
            opt.generations = integer(values, "--generations", opt.generations);
            opt.initialDepth = integer(values, "--depth", opt.initialDepth);
            opt.tournamentSize = integer(values, "--tournament", opt.tournamentSize);
            opt.crossoverProbability = decimal(values, "--crossover", opt.crossoverProbability);
            opt.maxLength = integer(values, "--max-length", opt.maxLength);
            opt.lambda = decimal(values, "--lambda", opt.lambda);
            opt.treeNorm = decimal(values, "--tree-norm", opt.treeNorm);
            opt.archiveBudget = integer(values, "--archive-budget", opt.archiveBudget);
            opt.validate();
            return opt;
        }

        private void validate() {
            if (populationSize < 2) throw new IllegalArgumentException("population must be >= 2");
            if (generations < 1) throw new IllegalArgumentException("generations must be >= 1");
            if (initialDepth < 0) throw new IllegalArgumentException("depth must be >= 0");
            if (tournamentSize < 1) throw new IllegalArgumentException("tournament must be >= 1");
            if (!(crossoverProbability >= 0 && crossoverProbability <= 1)) {
                throw new IllegalArgumentException("crossover must be in [0,1]");
            }
            if (maxLength < 3) throw new IllegalArgumentException("max-length must be >= 3");
            if (!(treeNorm > 0)) throw new IllegalArgumentException("tree-norm must be > 0");
            if (archiveBudget < 1) throw new IllegalArgumentException("archive-budget must be >= 1");
            if (window < 1 || replicate < 1) throw new IllegalArgumentException("window/replicate must be >= 1");
        }

        private static String required(Map<String, String> values, String key) {
            String value = values.get(key);
            if (value == null) throw new IllegalArgumentException("Missing required argument " + key);
            return value;
        }

        private static int integer(Map<String, String> values, String key, int defaultValue) {
            return values.containsKey(key) ? Integer.parseInt(values.get(key)) : defaultValue;
        }

        private static double decimal(Map<String, String> values, String key, double defaultValue) {
            return values.containsKey(key) ? Double.parseDouble(values.get(key)) : defaultValue;
        }

        private static void printUsage() {
            System.out.println("Required: --data FILE --candidates FILE --seed LONG --window N --replicate N");
            System.out.println("Optional: --population 500 --generations 200 --depth 5 --tournament 2");
            System.out.println("          --crossover 0.9 --max-length 10000");
            System.out.println("          --lambda 0.02 --tree-norm 100 --archive-budget 200");
        }
    }
}
