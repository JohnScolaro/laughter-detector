import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def histogram(data_list):
	x = np.array(data_list)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.set_axisbelow(True)

	# the histogram of the data
	n, bins, patches = plt.hist(x, 40, normed=1, facecolor='green', alpha=0.75, ec='black')

	plt.xlabel('Time (ms)')
	plt.ylabel('Frequency')
	plt.title('Histogram of Laughter Lengths')
	plt.grid(True, linestyle='--')
	plt.rcParams['grid.linestyle'] = "--"

	plt.show()

def main():
	# A counter to find total laugh count
	counter = 0
	# A counter to find total laugh length
	length = 0
	# A flag to determine whether it is the start or end of a laugh
	start_stop_flag = 0
	# Storage variables for the start and end of a laugh
	start_of_laugh = 0
	end_of_laugh = 0
	# Maximum laugh length
	max_laugh_length = 0
	# An array of lengths of laughter for stats
	laughter_list = []
	# Min and max laughs per file
	min_laughs_per_file = 9999999999 #Just some big number
	max_laughs_per_file = -1
	laughs_per_file = 0

	# For every file
	for f in list_of_files:

		# Create a file to open and write laughter times to
		new_file = open(f.rsplit('.', 1)[0] + ".ltimes", 'w');

		# Parse the XML file
		e = xml.etree.ElementTree.parse(f).getroot()
		# For the one timeorder section
		for timeorder in e.findall('TIME_ORDER'):
			# For each element in the timeorder section
			for timeslot in timeorder:
				if start_stop_flag == 0:
					# Set our flag to say that the next time is the end of the laugh
					start_stop_flag = 1
					# The laugh started at this time
					start_of_laugh = int(timeslot.attrib['TIME_VALUE'])
					# Write this to our new file
					new_file.write(str(start_of_laugh) + ",")
				else:
					# The next laugh time is the start of the next laugh
					start_stop_flag = 0
					# The laugh ended at this time
					end_of_laugh = int(timeslot.attrib['TIME_VALUE'])
					# Write this to our new file
					new_file.write(str(end_of_laugh) + "\n")
					# Incriment laugh count, find total laugh length and put in list
					counter += 1
					length += end_of_laugh - start_of_laugh
					laughter_list.append(end_of_laugh - start_of_laugh)
					# Is this the longest laugh?
					max_laugh_length = max(max_laugh_length, end_of_laugh - start_of_laugh)
					# File specific things
					laughs_per_file += 1

			# File specific things
			min_laughs_per_file = min(laughs_per_file, min_laughs_per_file)
			max_laughs_per_file = max(laughs_per_file, max_laughs_per_file)
			laughs_per_file = 0

			# Close our file
			new_file.close()

	# Print a bunch of fun facts
	print("Total laugh count: " + str(counter))
	print("Total length of laughter: " + str(length / 1000) + " seconds.")
	print("The longest laugh goes for: " + str(max_laugh_length / 1000) + " seconds.")
	print("The average length of a laughter segment is: {:.2f} milliseconds.".format(sum(laughter_list) / len(laughter_list)))
	print("The file with the least laughs has " + str(min_laughs_per_file) + " in it.")
	print("The file with the most laughs has " + str(max_laughs_per_file) + " in it.")
	histogram(laughter_list)

# Create our list of files
our_range = range(81)
list_of_files = []
for file_number in our_range:
	list_of_files.append("../Modified Audio/" + str(file_number + 1) + ".eaf")

main()
