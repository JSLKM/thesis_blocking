def clusterArray_to_blockDict(clusters):
	print("Hello")
	blocks = {}
	for index, value in enumerate(clusters):
		if value in blocks.keys():
			blocks[value].append(index)
		else:
			blocks[value] = list()
			blocks[value].append(index)
	return blocks