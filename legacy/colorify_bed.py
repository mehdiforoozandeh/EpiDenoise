import sys
import colorsys

def generate_colors(n):
    """
    Generate n visually distinct colors in RGB format.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Distribute hues evenly
        lightness = 0.5  # Medium lightness
        saturation = 0.9  # High saturation
        r_float, g_float, b_float = colorsys.hls_to_rgb(hue, lightness, saturation)
        r = int(r_float * 255)
        g = int(g_float * 255)
        b = int(b_float * 255)
        colors.append(f"{r},{g},{b}")
    return colors

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_bedgraph output_bed")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read input BEDGraph file and collect unique clusters
    cluster_set = set()
    bedgraph_data = []

    with open(input_file, 'r') as infile:
        for line in infile:
            if line.strip() == '':
                continue
            fields = line.strip().split('\t')
            if len(fields) < 4:
                print(f"Skipping malformed line: {line.strip()}", file=sys.stderr)
                continue
            chrom, start, end, cluster = fields[0], fields[1], fields[2], fields[3]
            cluster_set.add(cluster)
            bedgraph_data.append((chrom, start, end, cluster))

    # Generate colors for each unique cluster
    clusters = sorted(cluster_set)
    colors = generate_colors(len(clusters))
    cluster_color_map = dict(zip(clusters, colors))

    # Write output BED file with necessary columns
    with open(output_file, 'w') as outfile:
        for chrom, start, end, cluster in bedgraph_data:
            name = cluster  # Keep cluster ID as name
            score = '0'  # Set score to '0' or adjust as needed
            strand = '.'  # Set strand to '.'
            thickStart = start
            thickEnd = end
            itemRgb = cluster_color_map[cluster]
            outfile.write('\t'.join([chrom, start, end, name, score, strand, thickStart, thickEnd, itemRgb]) + '\n')

    print(f"Processed {len(bedgraph_data)} lines. Output written to {output_file}.")

if __name__ == "__main__":
    main()
