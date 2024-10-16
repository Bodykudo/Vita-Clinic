import { Request } from 'express';
import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  Query,
  Req,
  UnauthorizedException,
  UseGuards,
  ValidationPipe,
} from '@nestjs/common';

import { UsersService } from '../users.service';
import { PatientsService } from './patients.service';
import { AppointmentsService } from 'src/appointments/appointments.service';
import { ReportsService } from 'src/appointments/reports/reports.service';
import { ScansService } from 'src/appointments/scans/scans.service';
import { TreatmentService } from 'src/appointments/treatments/treatments.service';
import { PrescriptionsService } from 'src/appointments/prescriptions/prescriptions.service';
import { TestResultsService } from 'src/appointments/test-results/test-results.service';
import { JwtGuard } from 'src/auth/guards/jwt.guard';

import {
  GetAllUsersQuery,
  InsuranceDto,
  UpdateInsuranceDto,
} from '../dto/users.dto';
import { GetAllAppointmentsQuery } from 'src/appointments/dto/appointments.dto';
import { GetPatientReportsQuery } from 'src/appointments/reports/dto/reports.dto';
import { GetPatientTreatmentsQuery } from 'src/appointments/treatments/dto/treatments.dto';
import { GetPatientScansQuery } from 'src/appointments/scans/dto/scans.dto';
import { GetPatientPrescriptionsQuery } from 'src/appointments/prescriptions/dto/prescriptions.dto';
import { GetPatientTestResultsQuery } from 'src/appointments/test-results/dto/test-results.dto';
import type { Payload } from 'src/types/payload.type';

@Controller('/users/patients')
export class PatientsController {
  constructor(
    private readonly usersService: UsersService,
    private readonly patientsService: PatientsService,
    private readonly appointmentsService: AppointmentsService,
    private readonly reportsService: ReportsService,
    private readonly scansService: ScansService,
    private readonly treatmentService: TreatmentService,
    private readonly prescriptionsService: PrescriptionsService,
    private readonly testResultsService: TestResultsService,
  ) {}

  @UseGuards(JwtGuard)
  @Get()
  async getAllPatients(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetAllUsersQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.usersService.findAll(user.isSuperAdmin, 'patient', {
      ...query,
    });
  }

  @UseGuards(JwtGuard)
  @Get(':id/insurance')
  async getPatientInsurance(@Param('id') id: string, @Req() request: Request) {
    const user: Payload = request['user'];

    if (user.role === 'patient' && user.id !== id) {
      throw new UnauthorizedException();
    }

    return this.patientsService.getInsurance(id);
  }

  @UseGuards(JwtGuard)
  @Post(':id/insurance')
  async createPatientInsurance(
    @Param('id') id: string,
    @Req() request: Request,
    @Body(new ValidationPipe())
    dto: InsuranceDto,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'admin') {
      throw new UnauthorizedException();
    }

    return this.patientsService.craeteInsurance(id, dto, user.id);
  }

  @UseGuards(JwtGuard)
  @Patch(':id/insurance')
  async updatePatientInsurance(
    @Param('id') id: string,
    @Req() request: Request,
    @Body(new ValidationPipe())
    dto: UpdateInsuranceDto,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'admin') {
      throw new UnauthorizedException();
    }

    return this.patientsService.updateInsurance(id, dto, user.id);
  }

  @UseGuards(JwtGuard)
  @Delete(':id/insurance')
  async deletePatientInsurance(
    @Param('id') id: string,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (!user.isSuperAdmin) {
      throw new UnauthorizedException();
    }

    return this.patientsService.deleteInsurance(id, user.id);
  }

  @UseGuards(JwtGuard)
  @Get('/appointments')
  async getPatientsAppointments(
    @Query(new ValidationPipe({ transform: true }))
    query: GetAllAppointmentsQuery,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.findAll(query, user.id);
  }

  @UseGuards(JwtGuard)
  @Get(':id/appointments')
  async getPatientsAppointmentsById(
    @Param('id') id: string,
    @Query(new ValidationPipe({ transform: true }))
    query: GetAllAppointmentsQuery,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.appointmentsService.findAll(query, id);
  }

  @UseGuards(JwtGuard)
  @Get('/reports')
  async getPatientReports(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientReportsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.reportsService.findAllByPatientId(user.id, query);
  }

  @UseGuards(JwtGuard)
  @Get(':id/reports')
  async getPatientReportsById(
    @Param('id') id: string,
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientReportsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.reportsService.findAllByPatientId(id, query);
  }

  @UseGuards(JwtGuard)
  @Get('/scans')
  async getPatientScans(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientScansQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.scansService.findAllByPatientId(user.id, query);
  }

  @UseGuards(JwtGuard)
  @Get(':id/scans')
  async getPatientScansById(
    @Param('id') id: string,
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientReportsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.scansService.findAllByPatientId(id, query);
  }

  @UseGuards(JwtGuard)
  @Get('treatments')
  async getPatientTreatments(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientTreatmentsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.treatmentService.findAllByPatientId(user.id, query);
  }

  @UseGuards(JwtGuard)
  @Get(':id/treatments')
  async getPatientTreatmentsById(
    @Param('id') id: string,
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientTreatmentsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.treatmentService.findAllByPatientId(id, query);
  }

  @UseGuards(JwtGuard)
  @Get('prescriptions')
  async getPatientPrescriptions(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientPrescriptionsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.prescriptionsService.findAllByPatientId(user.id, query);
  }

  @UseGuards(JwtGuard)
  @Get(':id/prescriptions')
  async getPatientPrescriptionsById(
    @Param('id') id: string,
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientPrescriptionsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.prescriptionsService.findAllByPatientId(id, query);
  }

  @UseGuards(JwtGuard)
  @Get('test-results')
  async getPatientTestResults(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientTestResultsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'patient') {
      throw new UnauthorizedException();
    }

    return this.testResultsService.findAllByPatientId(user.id, query);
  }

  @UseGuards(JwtGuard)
  @Get(':id/test-results')
  async getPatientTestResultsById(
    @Param('id') id: string,
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetPatientTestResultsQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.testResultsService.findAllByPatientId(id, query);
  }
}
