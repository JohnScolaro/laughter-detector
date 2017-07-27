make list of file names															DONE
check names in list exist														DONE
	if not, remove																DONE
create reader																	DONE
decode csv																		DONE
	make the huge set of 1 dimensional arrays. Do it neatly.					DONE
stack individual cols of csv together to make tensor							DONE
make sure you start queue runners before attempting to do anything				DONE
change dataset creation so that it contains one file for all the data			DONE
get batches working																DONE
fix the input dimension size													DONE
get softmax_cross_entropy_with_logits working									DONE
stop queue runners																DONE
Get epochs working properly, and not just batches.								DONE - Could be better done though
	How are my batches even being produced? Is data repeated?					DONE - It turns out that tf.train.shuffle_batch selects individual lines randomly from random files. But it *does* do them with a uniform distribution. Using tf.train.batch does them all sequentially.
		Recreate using mock data and printing it through same pipeline.			DONE
	Is data contiguous? You should make it contiguous. (per batch)				DONE
Dump node weights																DONE
	Also, make pictures of node weights											DONE
	Also dump histograms of node values											DONE
Dump biases																		DONE
Dump cost at each epoch															DONE - but shit. Improve
	In the form where TensorBoard turns it into a pretty graph					DONE
Find a way to make a training set and test set.									DONE
Do something with queuerunners. Dump queue lengths if possible?					DONE - by accident
Improve data pipeline bottleneck												DONE
Create confusion matrix plots from accuracy evaluation							DONE
Why do TensorBoard graphs look shit if I don't delete after each run?			DONE
Why do TensorBoard pictures choose arb cycles to present in pictures?			DONE - Fixed with a new folder per run
Clean up python script. Put helpers in helper file.								DONE
Generalize the MLP creation functions so you can create any size MLP you want.	DONE
Improve performance.															DONE
Fix summary save file locations													DONE
Output time at each batch completion											DONE
Convert all datasets to TFRecords												DONE
design a TFRecord reader														DONE
rework input pipeline and file readers to use TFRecords							DONE
get subversion control working													DONE
no more feed dicts																DONE
further optimize																DONE
add name scopes to the mlp creation helper function								DONE
	Currently have shitty name scopes. Fix.										DONE
	Add a layer scope for each layers variables + ops.							NOT DONE - Seems they aren't actually used like this. Would require huge effort for no benefit.
refactor the two confusion and accuracy calculation functions in helpers		DONE
	By using streaming accuracy and confusion									DONE
	Fix streaming accuracy and conf, so that it is resetable.					DONE
Dump accuracy at each epoch														DONE
	Refactor input pipeline again												DONE
Calculate specificity and sensitivity											DONE
is accuracy actually being calculated correctly									DONE
Why is training slowing down?													MAYBE DONE
	Test if my small fix fixed it.												MAYBE DONE
Fix small normalized confusion matrix bug										DONE
normalize deltas and delta-deltas in the training set							DONE
play with hyper params															DONE
Add additional activation functions to the mlp model							DONE
find a way to fix the class imbalance problem.									DONE
	boost the amount of laughter in the dataset									PROBABLY WONT DO BECAUSE THE BELOW OPTION WORKS
	increase the weight of the laughter currently in the set					DONE
Make logger and the other summary logger use the same writer.					DONE
Change prints to print + log													DONE
Develop some more features														
Dump audio																		
	Dump laughter classified training audio										
	Dump laughter classified test audio											
make a real vs predicted laughter values graph in pyplot						DONE
Maybe test adding some normalization per batch?									
Fix summary writing bug? Kill tensorboard when starting training to fix.		MAYBE DONE?
Take the other column in the sound plot generator. The labels are opposite.		DONE
Change the plot size back to the default for the confusion matrix generator.	DONE
Make another program which runs net.py multiple times with different param		DONE
	Make it work well and give it a few tests.									DONE
	Clean it all up. Make helper functions to reduce code.
	Remove a few of the instances of code where you make folders if they dont exist.
Add dataset creation scripts to this repo										DONE
Make a "print_hyper_params" function.											TEST TO SEE IF DONE
Reroute stderr as well as stdout												DONE
Make sensitivity and specificity a class, so you can get max at end of run		DONE
Move confusion matrix maker to the plotters file								DONE - JUST TEST IT BEFORE CHECKING IT OFF
Attempt to make a rnn in another file											DONE
	Learn how rnns's work in tensorflow											DONE
		Funky input data requirements.											DONE
	Check out LTSM's while you're at it.										DONE
	Also add a MLP with multiple frames at once.								DONE
Add equal error rate, and other metrics to the metrics class?					
Try normalizing each layer of the mlp											
Try shuffling, and see if it helps.												
Fix bug in line 317 of helpers.py												DONE
Fiddle further with the initialization params
Double check the cost function actually works like you think it does.			
Get smoother output probabilities.												
Investigate adding a ROC curve when you get better looking outputs.				
Make sequence creation more efficient. (From stack overflow)
Change label handle creation in sequence creator to get the last element, not the middle.
Remember to add windows_length to net_runner if you want it to work.
Change input pipeline to use filters, and remove exception. (See github)
File by file mfcc normalization on the input.
