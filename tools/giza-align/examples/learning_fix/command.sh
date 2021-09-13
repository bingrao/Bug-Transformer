# https://sinaahmadi.github.io/posts/sentence-alignment-using-giza.html

./plain2snt.out ../giza_example/buggy.txt ../giza_example/fixed.txt
./snt2cooc.out ../giza_example/buggy.vcb ../giza_example/fixed.vcb ../giza_example/buggy_fixed.snt > ../giza_example/corp.cooc
./GIZA++ -S ../giza_example/buggy.vcb -T ../giza_example/fixed.vcb -C ../giza_example/buggy_fixed.snt -CoocurrenceFile ../giza_example/corp.cooc -outputpath ../giza_example/




./GIZA++-v2/plain2snt.out ./giza_example/buggy.txt ./giza_example/fixed.txt
./GIZA++-v2/snt2cooc.out  ./giza_example/buggy.vcb ./giza_example/fixed.vcb ./giza_example/buggy_fixed.snt >./giza_example/buggy_fixed.cooc
./GIZA++-v2/GIZA++ -S ./giza_example/buggy.vcb  -T ./giza_example/fixed.vcb  -C ./giza_example/buggy_fixed.snt -CoocurrenceFile ./giza_example/buggy_fixed.cooc -outputpath ./giza_example/output