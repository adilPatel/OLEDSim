CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lm

SRC = src/main.c src/config.c src/mesh.c src/dos.c src/mobility.c \
      src/recomb.c src/contacts.c src/newton.c src/exciton.c \
      src/outcoupling.c src/output.c src/util.c

oled_sim: $(SRC) src/oled.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
	rm -f oled_sim
	rm -rf output tests/out_*

test: oled_sim
	cd tests && ./run_tests.sh

.PHONY: clean test
