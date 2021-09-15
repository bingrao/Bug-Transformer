# https://sinaahmadi.github.io/posts/sentence-alignment-using-giza.html

./plain2snt.out ../examples/learning_fix/buggy.txt ../examples/learning_fix/fixed.txt
./snt2cooc.out ../examples/learning_fix/buggy.vcb ../examples/learning_fix/fixed.vcb ../examples/learning_fix/buggy_fixed.snt > ../examples/learning_fix/corp.cooc
./GIZA++ -S ../examples/learning_fix/buggy.vcb -T ../examples/learning_fix/fixed.vcb -C ../examples/learning_fix/buggy_fixed.snt -CoocurrenceFile ../examples/learning_fix/corp.cooc -outputpath ../examples/learning_fix/




./GIZA++-v2/plain2snt.out ./examples/learning_fix/buggy.txt ./examples/learning_fix/fixed.txt

# Feedward
./GIZA++-v2/snt2cooc.out  ./examples/learning_fix/buggy.vcb ./examples/learning_fix/fixed.vcb ./examples/learning_fix/buggy_fixed.snt >./examples/learning_fix/buggy_fixed.cooc
./GIZA++-v2/GIZA++ -S ./examples/learning_fix/buggy.vcb  -T ./examples/learning_fix/fixed.vcb  -C ./examples/learning_fix/buggy_fixed.snt -CoocurrenceFile ./examples/learning_fix/buggy_fixed.cooc -outputpath ./examples/learning_fix/output/feedward
python scripts/a3ToTalp.py < examples/learning_fix/output/feedward/2021-09-14.235320.bing.AA3.final > examples/learning_fix/output/feedward/2021-09-14.235320.bing.talp

# Backward
./GIZA++-v2/snt2cooc.out  ./examples/learning_fix/fixed.vcb ./examples/learning_fix/buggy.vcb ./examples/learning_fix/fixed_buggy.snt >./examples/learning_fix/fixed_buggy.cooc
./GIZA++-v2/GIZA++ -S ./examples/learning_fix/fixed.vcb  -T ./examples/learning_fix/buggy.vcb  -C ./examples/learning_fix/fixed_buggy.snt -CoocurrenceFile ./examples/learning_fix/fixed_buggy.cooc -outputpath ./examples/learning_fix/output/backward