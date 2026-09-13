package tui

import (
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"
)

type key uint8

const (
	keyUnknown key = iota
	keyUp
	keyDown
	keyLeft
	keyRight
	keyEnter
	keyEscape
	keyBackspace
	keyCtrlC
	keyResize
	keyCharacter
)

type keyEvent struct {
	key  key
	char byte
}

type input struct {
	bytes  chan byte
	errors chan error
	resize chan os.Signal
}

func newInput(reader io.Reader) *input {
	value := &input{
		bytes:  make(chan byte),
		errors: make(chan error, 1),
		resize: make(chan os.Signal, 1),
	}
	signal.Notify(value.resize, syscall.SIGWINCH)
	go func() {
		buffer := []byte{0}
		for {
			_, err := reader.Read(buffer)
			if err != nil {
				value.errors <- err
				return
			}
			value.bytes <- buffer[0]
		}
	}()
	return value
}

func (input *input) close() {
	signal.Stop(input.resize)
}

func (input *input) next() (keyEvent, error) {
	first, event, err := input.readByte(0)
	if event.key == keyResize || err != nil {
		return event, err
	}
	switch first {
	case 3:
		return keyEvent{key: keyCtrlC}, nil
	case 8, 127:
		return keyEvent{key: keyBackspace}, nil
	case '\r', '\n':
		return keyEvent{key: keyEnter}, nil
	case 27:
		second, event, err := input.readByte(35 * time.Millisecond)
		if event.key == keyResize || err != nil {
			return keyEvent{key: keyEscape}, nil
		}
		if second != '[' {
			return keyEvent{key: keyEscape}, nil
		}
		third, event, err := input.readByte(35 * time.Millisecond)
		if event.key == keyResize || err != nil {
			return keyEvent{key: keyEscape}, nil
		}
		switch third {
		case 'A':
			return keyEvent{key: keyUp}, nil
		case 'B':
			return keyEvent{key: keyDown}, nil
		case 'C':
			return keyEvent{key: keyRight}, nil
		case 'D':
			return keyEvent{key: keyLeft}, nil
		default:
			return keyEvent{key: keyUnknown}, nil
		}
	default:
		if first >= 32 && first <= 126 {
			return keyEvent{key: keyCharacter, char: first}, nil
		}
		return keyEvent{key: keyUnknown}, nil
	}
}

func (input *input) readByte(timeout time.Duration) (byte, keyEvent, error) {
	if timeout == 0 {
		select {
		case value := <-input.bytes:
			return value, keyEvent{}, nil
		case err := <-input.errors:
			return 0, keyEvent{}, err
		case <-input.resize:
			return 0, keyEvent{key: keyResize}, nil
		}
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case value := <-input.bytes:
		return value, keyEvent{}, nil
	case err := <-input.errors:
		return 0, keyEvent{}, err
	case <-input.resize:
		return 0, keyEvent{key: keyResize}, nil
	case <-timer.C:
		return 0, keyEvent{}, io.EOF
	}
}
