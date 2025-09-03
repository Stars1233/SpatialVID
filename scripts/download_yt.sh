
#!/bin/bash
# Batch download YouTube videos using a CSV list and record timing information.

csv_file_list=(
    "/path/to/video_csv/test-00000-of-00000.csv"
)

for csv_file in "${csv_file_list[@]}"; do
    start_time=$(date +%s)
    echo "-------------------- $csv_file --------------------"
    echo -e "\e[31mStart time: $start_time\e[0m"  # Red font
    python download.py --csv="$csv_file"
    end_time=$(date +%s)
    echo -e "\e[31mEnd time: $end_time\e[0m"    # Red font
    elapsed_time=$((end_time - start_time))
    echo -e "\e[32mElapsed time: ${elapsed_time}s\e[0m"  # Green font
done

# Quick command: count number of downloaded JSON files
# ls -l videos/patch1_sample_100_1/*.json | wc -l

# Check available formats for a YouTube video
# yt-dlp -F --list-formats "https://www.youtube.com/watch?v=omP01s7RUSA" --proxy 127.0.0.1:7892 --cookies cookies.txt