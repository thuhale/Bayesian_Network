import java.util.Map;
import java.util.HashMap;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	private final double delta = 0.00001;
	
	private Map<Label, Integer> labelCount;
	private Map<Label, Map<String, Integer> > wordTypeCount;
	private Map<Label, Integer> labelTokenCount;
	private int numInstance;
	//private int numTotalToken;
	private int vocabularySize;
	
	
	@Override
	public void train(Instance[] trainingData, int v) {
		//Implement
		//Update labelCount and labelTokenCount
		numInstance = trainingData.length;
		vocabularySize = v;
		
		
		int label_ham_count = 0;
		int label_spam_count = 0;
		int token_ham_count = 0;
		int token_spam_count = 0;
		labelCount = new HashMap<Label, Integer>();
		labelTokenCount = new HashMap<Label, Integer>();
		wordTypeCount =  new HashMap<Label, Map<String, Integer>>();
		Map<String, Integer> ham_wordTypeCount = new HashMap<String, Integer>();
		Map<String, Integer> spam_wordTypeCount = new HashMap<String, Integer>();
		
		
		
		
		
		for (int i = 0; i < trainingData.length; i++){
			Instance inst = trainingData[i];
			if(inst.label == Label.HAM){
				label_ham_count++;
				token_ham_count = token_ham_count+inst.words.length;
				for(int j = 0; j < inst.words.length; j++){
					if(ham_wordTypeCount.containsKey(inst.words[j])){
						int temp = ham_wordTypeCount.get(inst.words[j]);
						temp = temp+1;
						ham_wordTypeCount.put(inst.words[j], temp);
					}
					else{
						ham_wordTypeCount.put(inst.words[j], new Integer(1));
					}
				}
			}
			else{
				label_spam_count++;
				token_spam_count = token_spam_count+inst.words.length;
				for(int j = 0; j < inst.words.length; j++){
					if(spam_wordTypeCount.containsKey(inst.words[j])){
						int temp = spam_wordTypeCount.get(inst.words[j]);
						temp = temp+1;
						spam_wordTypeCount.put(inst.words[j], temp);
					}
					else{
						spam_wordTypeCount.put(inst.words[j], new Integer(1));
					}
				}
			}
			
		}
		labelCount.put(Label.HAM, label_ham_count);
		labelCount.put(Label.SPAM, label_spam_count);
		
		labelTokenCount.put(Label.HAM, token_ham_count);
		labelTokenCount.put(Label.SPAM, token_spam_count);	
		
		wordTypeCount.put(Label.HAM, ham_wordTypeCount );
		wordTypeCount.put(Label.SPAM,spam_wordTypeCount );
		
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {				
		//Implement
		//Calculate prior probability using training data
		
		double prior_prob = 0.0;
		
		if(label == Label.HAM){
			prior_prob = this.labelCount.get(Label.HAM)*1.0/(this.labelCount.get(Label.HAM)*1.0+this.labelCount.get(Label.SPAM)*1.0);			
		}
		else{
			prior_prob = this.labelCount.get(Label.SPAM)*1.0/(this.labelCount.get(Label.HAM)*1.0+this.labelCount.get(Label.SPAM)*1.0);			
		}
		return prior_prob;
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		// Implement
		double conditional_likelihood = 0.0;
		if(label == Label.HAM){
			int ham_word_count = 0;
			if(wordTypeCount.get(Label.HAM).containsKey(word)){
				ham_word_count = wordTypeCount.get(Label.HAM).get(word);
			}
			conditional_likelihood = (ham_word_count*1.0+delta)/(vocabularySize*1.0*delta+labelTokenCount.get(Label.HAM));
		}
		else{
			int spam_word_count = 0;
			if(wordTypeCount.get(Label.SPAM).containsKey(word)){
				spam_word_count = wordTypeCount.get(Label.SPAM).get(word);
			}
			conditional_likelihood = (spam_word_count*1.0+delta)/(vocabularySize*1.0*delta+labelTokenCount.get(Label.SPAM));
		}
	
		return conditional_likelihood;
	}
	
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		// Implement
		ClassifyResult classify = new ClassifyResult();
		classify.log_prob_ham = Math.log(this.p_l(Label.HAM));
		classify.log_prob_spam = Math.log(this.p_l(Label.SPAM));
		
		for(int i = 0; i < words.length; i++){
			classify.log_prob_ham = classify.log_prob_ham + Math.log(this.p_w_given_l(words[i], Label.HAM));
			classify.log_prob_spam = classify.log_prob_spam + Math.log(this.p_w_given_l(words[i], Label.SPAM));
		}
		
		if (classify.log_prob_ham >= classify.log_prob_spam){
			classify.label = Label.HAM;
		}
		else{
			classify.label = Label.SPAM;
		}
		
		
		return classify;
	}
	
	/**
	 * Gets the confusion matrix for a test set. 
	 */
	@Override
	public ConfusionMatrix calculateConfusionMatrix(Instance[] testData)
	{
		// Implement
		int truePositives = 0;
		int trueNegatives = 0;
		int falsePositives = 0;
		int falseNegatives = 0;
		
		for(int i = 0; i < testData.length; i++){
			Instance inst = testData[i];
			ClassifyResult classify = this.classify(inst.words);
			if (classify.label == Label.SPAM && inst.label == Label.SPAM){
				truePositives++;
			}
			else if (classify.label == Label.SPAM && inst.label == Label.HAM){
				falsePositives++;
			}
			else if (classify.label == Label.HAM && inst.label == Label.HAM){
				trueNegatives++;
			}
			else{
				falseNegatives++;
			}
			
		}
		ConfusionMatrix cm = new ConfusionMatrix(truePositives,trueNegatives, falsePositives , falseNegatives);
		
		
		return cm;
	}
}
