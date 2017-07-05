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
add name scopes to the mlp creation helper function
refactor the two confusion and accuracy calculation functions in helpers
Dump accuracy at each epoch
Dump audio
	Dump laughter classified training audio
	Dump laughter classified test audio
Insert arbitrary images into TensorBoard
	Embed confusion matrix plots into TensorBoard
Why do pictures in tensorboard only have 10 images and not the full amount?