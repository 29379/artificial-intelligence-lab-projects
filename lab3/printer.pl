% instruction
% file:///D:/studia/6%20semestr/Sztuczna%20inteligencja%20i%20in%C5%BCynieria%20wiedzy/LAB/SIiIW%20lab3.pdf

% manuals
% https://download.brother.com/welcome/doc003208/cv_dcp100_pol_busr.pdf
% https://download.brother.com/welcome/doc003209/cv_dcp100_pol_ausr.pdf
% https://download.brother.com/welcome/doc003205/cv_dcp100_pol_qsg_let122037.pdf

% Printer facts
printer(brother_dcp_j105).
printer_type(brother_dcp_j105, inkjet).

print_function(brother_dcp_j105, black_and_white).
print_function(brother_dcp_j105, color).
scan_function(brother_dcp_j105, black_and_white).
scan_function(brother_dcp_j105, color).
copy_function(brother_dcp_j105, black_and_white).
copy_function(brother_dcp_j105, color).

% Ink facts
ink_cartridge(brother_dcp_j105, black).
ink_cartridge(brother_dcp_j105, cyan).
ink_cartridge(brother_dcp_j105, magenta).
ink_cartridge(brother_dcp_j105, yellow).

% Paper handling facts
paper_size_supported(brother_dcp_j105, a4).
paper_size_supported(brother_dcp_j105, a5).
paper_size_supported(brother_dcp_j105, letter).
paper_size_supported(brother_dcp_j105, photo10x15).

% Connectivity facts
connectivity(brother_dcp_j105, usb).
connectivity(brother_dcp_j105, wifi).

% Power supply facts
power_supply(brother_dcp_j105, electric).
power_voltage(brother_dcp_j105, 220).

% Additional features facts
has_fax(brother_dcp_j105, no).
has_duplex_printing(brother_dcp_j105, no).
has_card_reader(brother_dcp_j105, no).

% Physical elements facts
printer_part(power_cable).
printer_part(usb_cable).
printer_part(black_ink_cartridge).
printer_part(cyan_ink_cartridge).
printer_part(magenta_ink_cartridge).
printer_part(yellow_ink_cartridge).
printer_part(paper_tray).
printer_part(paper).
printer_part(printhead).


% Panel elements facts
printer_part(power_button).
printer_part(start_mono_button).
printer_part(start_color_button).
printer_part(stop_button).
printer_part(scan_button).
printer_part(menu_button).
printer_part(arrow_up_button).
printer_part(ok_button).
printer_part(arrow_down_button).
printer_part(enlarge_shrink_button).
printer_part(copy_quality_button).
printer_part(copy_quantity_button).
printer_part(copy_options_button).

printer_part(warning_light).
printer_part(lcd_screen).

% ------------------------------------------

% Warning light and lcd screen states, that
% represent different states of the printer
% State ---> State | Warning_light state | Lcd_screen state)

printer_state(ok, off, 'Urządzenie DCP jest gotowe do użycia').

printer_state(not_plugged_in, off, off).
printer_state(turned_off, off, off).

printer_state(no_paper, orange, 'Brak papieru').
printer_state(paper_jam, orange, 'Zator papieru').
printer_state(paper_size, orange, 'Zły rozmiar papieru').
printer_state(cartridge_not_installed_correctly, orange, 'Brak wkładu atr').
printer_state(ink_not_detected, orange, 'Nie można wykryć').

printer_state(no_ink_black, orange, 'Druk niemożliwy. Wymień tusz czarny').
printer_state(no_ink_cyan, orange, 'Druk niemożliwy. Wymień tusz cyan').
printer_state(no_ink_magenta, orange, 'Druk niemożliwy. Wymień tusz magenta').
printer_state(no_ink_yellow, orange, 'Druk niemożliwy. Wymień tusz yellow').
printer_state(low_on_black, orange, 'Mało czarny').
printer_state(low_on_cyan, orange, 'Mało cyan').
printer_state(low_on_magenta, orange, 'Mało magenta').
printer_state(low_on_yellow, orange, 'Mało żółty').

% printer_state(out_of_memory, orange, 'Pamięć urządzenia jest pełna').
% printer_state(mechanical, orange, 'Nie moż. ... X').
% printer_state(low_temp, orange, 'Niska temperatura').
% printer_state(almost_full_absorber, orange, 'Poch.atr.pr.peł.').
% printer_state(absorber_full, orange, 'Pochł.atr. pełny').
% printer_state(open_cover, orange, 'Pokrywa otwarta').
% printer_state(data_left_out, orange, 'Pozostałe dane').
% printer_state(mono_only, orange, 'Tylko druk mono. Wymień tusz X').
% printer_state(high_temp, orange, 'Wysoka temperat.').
% printer_state(ink_cover, orange, 'Zam. pokr. tuszu').

% ------------------------------------------
% Problem ---> Name | State

problem(no_power, not_plugged_in).
problem(printer_off, turned_off).

problem(unable_to_print, paper_jam).
problem(unable_to_print, no_paper).
problem(unable_to_print, paper_size).
problem(unable_to_print, ink_not_detected).
problem(unable_to_print, cartridge_not_installed_correctly).

problem(unable_to_print, no_ink_black).
problem(unable_to_print, no_ink_cyan).
problem(unable_to_print, no_ink_magenta).
problem(unable_to_print, no_ink_yellow).

problem(print_low_color, low_on_black).
problem(print_low_color, low_on_cyan).
problem(print_low_color, low_on_magenta).
problem(print_low_color, low_on_yellow).

% ------------------------------------------
% Cause ---> Printer part | Problem | Printer state

cause(power_cable, no_power, not_plugged_in).
cause(power_button, printer_off, turned_off).

cause(paper, unable_to_print, no_paper).
cause(paper, unable_to_print, paper_jam).
cause(paper, unable_to_print, paper_size).

cause(black_ink_cartridge, unable_to_print, no_ink_black).
cause(cyan_ink_cartridge, unable_to_print, no_ink_cyan).
cause(magenta_ink_cartridge, unable_to_print, no_ink_magenta).
cause(yellow_ink_cartridge, unable_to_print, no_ink_yellow).

cause(black_ink_cartridge,print_low_color, low_on_black).
cause(cyan_ink_cartridge,print_low_color, low_on_cyan).
cause(magenta_ink_cartridge,print_low_color, low_on_magenta).
cause(yellow_ink_cartridge,print_low_color, low_on_yellow).

% ------------------------------------------
% Fix ---> Printer state | string(short explanation)

fix(not_plugged_in, 'Insert a power cable').
fix(turned_off, 'Press the "Power" button').

fix(no_paper, 'Insert paper into the paper tray').
fix(paper_jam, 'Pull out the jammed paper').
fix(paper_size, 'Insert paper in the proper size').

fix(no_ink_black, 'Replace the ink inside the black ink cartridge').
fix(no_ink_cyan, 'Replace the ink inside the cyan ink cartridge').
fix(no_ink_magenta, 'Replace the ink inside the magenta ink cartridge').
fix(no_ink_yellow, 'Replace the ink inside the yellow ink cartridge').

fix(low_on_black, 'Replace the ink inside the black ink cartridge').
fix(low_on_cyan, 'Replace the ink inside the cyan ink cartridge').
fix(low_on_magenta, 'Replace the ink inside the magenta ink cartridge').
fix(low_on_yellow, 'Replace the ink inside the yellow ink cartridge').

% ------------------------------------------
% Basic queries - commented out to avoid errors

% Retrieve all the supported printing functions
%print_function(brother_dcp_j105, Function).

% Retrieve the printer state, warning light color and message for each of the states
%printer_state(State, WarningLight, Message).

% Retrieve all the ink cartridges supported by the printer
%ink_cartridge(brother_dcp_j105, Cartridge).

% Find all the paper sizes supported by the printer
%paper_size_supported(brother_dcp_j105, Size).

% Check if the printer supports a fax feature
%has_fax(brother_dcp_j105, Feature).

% Find all the physical parts of a printer
%printer_part(Part).

% Check if the printer supports duplex printing
%has_duplex_printing(brother_dcp_j105, Feature).

% ---------------------------------------------
% More complex queries

find_states_for_problem(Problem) :-
    printer_state(State, DiodeState, Desc),
    (problem(Problem, State); cause(_, Problem, State)),
    write('Stan: '), write(State), nl,
    write('Stan diody: '), write(DiodeState), nl,
    write('Opis: '), write(Desc), nl,
    nl,
    fail.

find_parts_and_states_for_problem(Problem) :-
    cause(Part, Problem, State),
    printer_state(State, _, Desc),
    write('Część drukarki: '), write(Part), nl,
    write('Stan: '), write(State), nl,
    write('Opis: '), write(Desc), nl,
    nl,
    fail.

find_solutions_for_problem(Problem) :-
    fix(State, Desc),
    problem(Problem, State),
    write('Stan: '), write(State), nl,
    write('Opis rozwiązania: '), write(Desc), nl,
    nl,
    fail.

find_possible_issues_for_part(Part) :-
    cause(Part, Problem, State),
    write('Część drukarki: '), write(Part), nl,
    write('Problem: '), write(Problem), nl,
    write('Stan drukarki: '), write(State), nl,
    nl,
    fail.