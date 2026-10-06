# Usage: ./download_data.sh [out_dir]
# Downloads INCLUDE (Zenodo record 4010759) into out_dir (default: current directory)
# and extracts it into out_dir/original, the layout expected by format.py
set -e
mkdir -p "${1:-.}"
cd "${1:-.}"

for i in {1..8}
do
    wget https://zenodo.org/record/4010759/files/Adjectives_${i}of8.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Animals_${i}of2.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Clothes_${i}of2.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Colours_${i}of2.zip
done

for i in {1..3}
do
    wget https://zenodo.org/record/4010759/files/Days_and_Time_${i}of3.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Electronics_${i}of2.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Greetings_${i}of2.zip
done

for i in {1..4}
do
    wget https://zenodo.org/record/4010759/files/Home_${i}of4.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Jobs_${i}of2.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Means_of_Transportation_${i}of2.zip
done

for i in {1..5}
do
    wget https://zenodo.org/record/4010759/files/People_${i}of5.zip
done

for i in {1..4}
do
    wget https://zenodo.org/record/4010759/files/Places_${i}of4.zip
done

for i in {1..2}
do
    wget https://zenodo.org/record/4010759/files/Pronouns_${i}of2.zip
done

wget https://zenodo.org/record/4010759/files/Seasons_1of1.zip

for i in {1..3}
do
    wget https://zenodo.org/record/4010759/files/Society_${i}of3.zip
done

# unzip all files
for f in *.zip
do
    unzip -o "$f" -d original
done